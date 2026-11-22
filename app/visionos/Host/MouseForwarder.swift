import Foundation
import GameController

/// Feeds a paired mouse to the game while it is shown in the immersive
/// space, where no SDL window has focus and so no pointer events reach SDL.
///
/// GameController reports the mouse as raw deltas and buttons, which go to
/// the runtime's host input entries (runtime/host.h); the runtime turns
/// them into the game's own messages, keeping a virtual pointer for menus.
/// The keyboard needs nothing here: SDL already reads `GCKeyboard` itself,
/// and with the host's claim on the runtime accepts those keys without a
/// focused window. Gamepads come through SDL's gamepad layer the same way.
///
/// SDL's own `GCMouse` handlers stay off because the app does not declare
/// `UIApplicationSupportsIndirectInputEvents`; adding that key would make
/// SDL install handlers on the same mice and double every wheel notch.
///
/// Everything here is nonisolated: the handlers run on `queue`, and a
/// closure written in main actor code would be checked as main actor
/// isolated when GameController calls it there, which traps.
final class MouseForwarder: @unchecked Sendable {
    nonisolated static let shared = MouseForwarder()

    private let queue = DispatchQueue(label: "host.mouse", qos: .userInteractive)
    private var observers: [NSObjectProtocol] = []
    private var started = false

    /// Hooks every mouse connected now and later. Called once, when the
    /// immersive space opens; the window path never calls it.
    nonisolated func start() {
        queue.async { [self] in
            guard !started else { return }
            started = true
            let center = NotificationCenter.default
            observers.append(
                center.addObserver(forName: .GCMouseDidConnect, object: nil, queue: nil) { [self] note in
                    // Hooked where the notification arrives: setting a
                    // mouse's handlers is safe from any thread.
                    guard let mouse = note.object as? GCMouse else { return }
                    self.hook(mouse)
                }
            )
            for mouse in GCMouse.mice() {
                hook(mouse)
            }
            ScreenCompositor.log("mouse input from GameController, \(GCMouse.mice().count) connected")
        }
    }

    private func hook(_ mouse: GCMouse) {
        guard let input = mouse.mouseInput else { return }
        mouse.handlerQueue = queue
        input.mouseMovedHandler = { _, deltaX, deltaY in
            // GameController's y grows upwards; the game's grows down.
            rt_host_mouse_motion(deltaX, -deltaY)
        }
        input.leftButton.pressedChangedHandler = { _, _, pressed in
            rt_host_mouse_button(1, pressed ? 1 : 0)
        }
        input.middleButton?.pressedChangedHandler = { _, _, pressed in
            rt_host_mouse_button(2, pressed ? 1 : 0)
        }
        input.rightButton?.pressedChangedHandler = { _, _, pressed in
            rt_host_mouse_button(3, pressed ? 1 : 0)
        }
        for (index, button) in (input.auxiliaryButtons ?? []).prefix(2).enumerated() {
            let number = Int32(4 + index)
            button.pressedChangedHandler = { _, _, pressed in
                rt_host_mouse_button(number, pressed ? 1 : 0)
            }
        }
        input.scroll.valueChangedHandler = { _, xValue, _ in
            // The vertical wheel arrives on the first axis, negative when
            // rolled away from the user, as SDL's own handler reads it.
            if xValue != 0 { rt_host_mouse_wheel(-xValue) }
        }
        ScreenCompositor.log("mouse hooked: \(mouse.vendorName ?? "unnamed")")
    }
}
