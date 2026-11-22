import Foundation
import GameController

/// Logs GameController's own view of the game controllers to the run log,
/// as ground truth beside what SDL and the runtime's pad layer make of it
/// ("pad added", "pad combined", "pad drives").
///
/// SDL reads controllers through GameController as well, and its connect
/// and disconnect observers run wherever the framework posts, so they do
/// not depend on SDL's event pump, which is off here. But SDL takes a
/// controller's element list once, when it connects
/// (`IOS_AddMFIJoystickDevice` in SDL_mfijoystick.m). A PSVR2 Sense half
/// that connects on its own is a controller of its own, "(L)" or "(R)",
/// and both halves on at launch are one "(L/R)" controller. What the
/// framework does when the second half joins a lone first one is not
/// documented, so every connect, disconnect and change of current
/// controller is logged with its name, product category and elements, and
/// the list is checked every two seconds for a controller that changed its
/// name or elements without a new connect.
///
/// A controller that did change in place would leave SDL reading the
/// elements it saw at connect, so a left stick that arrived later would
/// never reach the game. Such a controller is announced again: a
/// disconnect and a connect posted for the same object, which SDL's
/// observers answer by dropping it and adding it back with its elements as
/// they are now. The pad layer sees that as a pad removed and added. It is
/// done at most a few times per controller, so a controller whose elements
/// kept changing could not make the pad flap for the whole session.
///
/// Spatial accessories are logged too, since visionOS 27 can present
/// tracked controllers that way as well, and one that never shows up as a
/// `GCController` would explain a controller SDL never sees.
///
/// Everything runs on the main thread: the observers are queued on the
/// main operation queue and the check is a main queue timer, so the state
/// here needs no lock. The class is nonisolated so its closures are not
/// checked as main actor closures when the framework calls them.
final class ControllerWatcher: @unchecked Sendable {
    nonisolated static let shared = ControllerWatcher()

    private struct Seen: Equatable {
        var name: String
        var category: String
        var elements: [String]
    }

    private static let checkInterval = DispatchTimeInterval.seconds(2)
    private static let maxReannounces = 3

    private var observers: [NSObjectProtocol] = []
    private var timer: DispatchSourceTimer?
    private var seen: [ObjectIdentifier: Seen] = [:]
    private var reannounced: [ObjectIdentifier: Int] = [:]
    /// Notifications this class posted itself and has not yet received
    /// back, so the log can tell them from the framework's own.
    private var echoes: [ObjectIdentifier: Int] = [:]
    private var started = false
    private var listed = false
    private var lastCurrent: ObjectIdentifier?
    private var currentLines = 0
    private var gripWired = Set<ObjectIdentifier>()
    /// Each Sense half's Grip, read on a timer: its change handler never
    /// fires (headset run, 2026-11-21), as the other port read its halves'
    /// state every tick rather than wait for one.
    private var grips: [(button: GCPressedStateInput, side: Int32)] = []
    private var gripDown: [Bool] = [false, false]
    private var gripLines = 0
    private var gripTimer: Timer?

    /// Called on the main thread once the runtime has started, so the
    /// lines land in its log. The first listing waits for the check, by
    /// when the runtime has opened its log.
    nonisolated func start() {
        guard !started else { return }
        started = true
        let center = NotificationCenter.default
        let events: [(NSNotification.Name, String)] = [
            (.GCControllerDidConnect, "connected"),
            (.GCControllerDidDisconnect, "disconnected"),
            (.GCControllerDidBecomeCurrent, "became current"),
        ]
        for (name, event) in events {
            observers.append(
                center.addObserver(forName: name, object: nil, queue: .main) { [self] note in
                    guard let controller = note.object as? GCController else { return }
                    self.report(controller, event)
                }
            )
        }
        let accessoryEvents: [(NSNotification.Name, String)] = [
            (.GCSpatialAccessoryDidConnect, "connected"),
            (.GCSpatialAccessoryDidDisconnect, "disconnected"),
        ]
        for (name, event) in accessoryEvents {
            observers.append(
                center.addObserver(forName: name, object: nil, queue: .main) { note in
                    guard let accessory = note.object as? GCSpatialAccessory else { return }
                    Self.log("spatial accessory \(event): \(Self.label(accessory))")
                }
            )
        }
        let timer = DispatchSource.makeTimerSource(queue: .main)
        timer.schedule(deadline: .now() + .seconds(1), repeating: Self.checkInterval)
        timer.setEventHandler { [self] in
            self.check()
        }
        timer.resume()
        self.timer = timer
    }

    private func report(_ controller: GCController, _ event: String) {
        let id = ObjectIdentifier(controller)
        let now = Self.snapshot(controller)
        if event == "became current" {
            // Every switch of hands between two halves makes the other one
            // current, so this line is the name alone, once per switch, and
            // only the first few: with both halves in use it is every press.
            if lastCurrent != id {
                lastCurrent = id
                currentLines += 1
                if currentLines <= 20 {
                    Self.log("controller became current: \(now.name)\(currentLines == 20 ? " (no more of these)" : "")")
                }
            }
            // Becoming current is no connect, so it must not hide a change
            // the check would otherwise catch.
            if let before = seen[id], before != now {
                changed(controller, id, before, now)
            } else {
                seen[id] = now
            }
            return
        }
        var line = "controller \(event): \(Self.describe(now))"
        var echo = false
        if let pending = echoes[id], pending > 0 {
            echoes[id] = pending - 1
            echo = true
            line += " (announced again for SDL)"
        }
        Self.log(line)
        if event == "disconnected" {
            seen[id] = nil
            if lastCurrent == id { lastCurrent = nil }
            // A real disconnect ends this controller; a later one at the
            // same address starts its count afresh.
            if !echo { reannounced[id] = nil }
        } else {
            seen[id] = now
        }
    }

    private func check() {
        let controllers = GCController.controllers()
        if !listed {
            listed = true
            Self.log("controllers at start: \(controllers.count)")
            let accessories = GCSpatialAccessory.spatialAccessories
            Self.log("spatial accessories at start: \(accessories.count)")
            for accessory in accessories {
                Self.log("spatial accessory present: \(Self.label(accessory))")
            }
        }
        var present = Set<ObjectIdentifier>()
        for controller in controllers {
            let id = ObjectIdentifier(controller)
            present.insert(id)
            if !gripWired.contains(id) { wireGrip(controller, id) }
            let now = Self.snapshot(controller)
            guard let before = seen[id] else {
                Self.log("controller present: \(Self.describe(now))")
                seen[id] = now
                continue
            }
            if before != now {
                changed(controller, id, before, now)
            }
        }
        gripWired.formIntersection(present)
        for id in seen.keys where !present.contains(id) {
            Self.log("controller gone without a disconnect: \(Self.describe(seen[id]!))")
            seen[id] = nil
        }
    }

    private func changed(_ controller: GCController, _ id: ObjectIdentifier, _ before: Seen, _ now: Seen) {
        seen[id] = now
        Self.log("controller changed in place: was \(Self.describe(before)); now \(Self.describe(now))")
        reannounce(controller, id)
    }

    private func reannounce(_ controller: GCController, _ id: ObjectIdentifier) {
        let count = reannounced[id, default: 0]
        guard count < Self.maxReannounces else {
            Self.log("controller changed again; not announced again, \(count) times already")
            return
        }
        reannounced[id] = count + 1
        echoes[id, default: 0] += 2
        let center = NotificationCenter.default
        center.post(name: .GCControllerDidDisconnect, object: controller)
        center.post(name: .GCControllerDidConnect, object: controller)
    }

    /// A Sense half reports its shoulder as Grip, which the pad layer's SDL
    /// driver leaves unmapped, so L1 and R1 did nothing once the halves
    /// came apart. The Grip goes to the runtime as that half's shoulder.
    private func wireGrip(_ controller: GCController, _ id: ObjectIdentifier) {
        gripWired.insert(id)
        let name = controller.vendorName ?? ""
        let side: Int32
        if name.hasSuffix("(L)") {
            side = 0
        } else if name.hasSuffix("(R)") {
            side = 1
        } else {
            return
        }
        // Through the controller's input, not its profile: on a Sense half
        // the profile's Grip never changes (headset run, 2026-11-21), as
        // its left stick reads 0 there.
        guard let grip = controller.input.buttons[.grip] else {
            Self.log("no Grip on \(name)")
            return
        }
        grips.removeAll { $0.side == side }
        grips.append((grip.pressedInput, side))
        if gripTimer == nil {
            gripTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 120.0, repeats: true) { [weak self] _ in
                MainActor.assumeIsolated { self?.pollGrips() }
            }
        }
        Self.log("Grip on \(name) is its shoulder button")
    }

    private func pollGrips() {
        for g in grips {
            let down = g.button.isPressed
            let i = Int(g.side)
            guard down != gripDown[i] else { continue }
            gripDown[i] = down
            rt_host_pad_grip(g.side, down ? 1 : 0)
            if gripLines < 6 {
                gripLines += 1
                Self.log("Grip \(i == 0 ? "left" : "right") \(down ? "down" : "up")")
            }
        }
    }

    private static func snapshot(_ controller: GCController) -> Seen {
        Seen(
            name: controller.vendorName ?? "unnamed",
            category: controller.productCategory,
            elements: controller.physicalInputProfile.elements.keys.sorted()
        )
    }

    private static func describe(_ seen: Seen) -> String {
        "\(seen.name) [\(seen.category)] \(seen.elements.count) elements: \(seen.elements.joined(separator: ", "))"
    }

    private static func label(_ accessory: GCSpatialAccessory) -> String {
        "\(accessory.vendorName ?? "unnamed") [\(accessory.productCategory)]"
    }

    private static func log(_ message: String) {
        rt_host_log(message)
    }
}
