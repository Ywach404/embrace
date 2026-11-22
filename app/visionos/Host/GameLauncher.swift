import Foundation
import Observation
import UIKit

/// Finds the player's game data and starts the runtime on its own thread.
///
/// The game data lives in `Documents/VTMB`, which the Files app shows
/// because the app shares its Documents folder; the app bundle never holds
/// any of it. Everything the runtime writes (settings, saves, logs, the
/// shader and pipeline caches) goes to the overlay in Application Support,
/// so the player's copy is only ever read.
@MainActor
@Observable
final class GameLauncher {
    enum State: Equatable {
        case idle
        case missingData
        case running
        /// The runtime's main returned. `reason` is the refusal the runtime
        /// gave, or nil when it returned without one.
        case stopped(status: Int32, reason: String?)
        case failedToStart(Int32)
    }

    /// The desktop the runtime is told it has. The headset has none to
    /// measure, and the runtime fits a 4:3 window into the usable area, so a
    /// 4:3 size here gives a game window of exactly this size. Full view
    /// draws the headset's whole field instead, at the shape of the union of
    /// both eyes' views (about 1.45 across to 1), and the runtime is asked
    /// for that size with `--aspect`; `RT_FULLVIEW_SIZE=WxH` changes it
    /// (2048x1408 is a lighter size).
    nonisolated static let desktopSize: (width: Int, height: Int) = wantsFullView
        ? fullViewSize : (width: 1402, height: 1052)

    nonisolated private static var fullViewSize: (width: Int, height: Int) {
        let parts = (ProcessInfo.processInfo.environment["RT_FULLVIEW_SIZE"] ?? "")
            .split(separator: "x").compactMap { Int($0) }
        if parts.count == 2, (640...4096).contains(parts[0]), (480...4096).contains(parts[1]) {
            return (width: parts[0], height: parts[1])
        }
        // 1.125 times 2048x1408 (1.27 times the pixels).
        // That run spent 9 ms of GPU per game frame of an 11.6 ms budget at
        // 86 Hz, and each eye still sampled the image at about 0.7 of the
        // drawable's density, which read as grain; this is the step the
        // headroom allows. 2560x1760 or more needs a GPU check first.
        return (width: 2304, height: 1584)
    }

    /// `--immersive` among the app's launch arguments shows the game on a
    /// virtual screen in an immersive space instead of only in the window.
    /// It is the host's option, taken out before the runtime sees the
    /// arguments, since the runtime refuses options it does not know.
    nonisolated static let immersiveFlag = "--immersive"
    nonisolated static let isImmersive = ProcessInfo.processInfo.arguments.dropFirst().contains(immersiveFlag)
        || wantsFullView
    nonisolated static let immersiveSpaceID = "screen"

    /// Full view, the default: play frames fill each eye's whole view from
    /// that eye's own position, with the head driving the camera
    /// (metal_d3d9_fullview_enable); menus come on stereo's plane. It
    /// implies `--immersive`, `--stereo` and head look. A launch from the
    /// Home View has no arguments, so it is on unless `--nofullview` or
    /// `RT_FULLVIEW=0` asks for the window (or, with `--immersive`, the
    /// theater screen). `--fullview` is still taken. The runtime sees none
    /// of these.
    nonisolated static let fullViewFlag = "--fullview"
    nonisolated static let noFullViewFlag = "--nofullview"
    nonisolated static let wantsFullView = !ProcessInfo.processInfo.arguments.dropFirst().contains(noFullViewFlag)
        && ProcessInfo.processInfo.environment["RT_FULLVIEW"] != "0"

    /// `--stereo`, on top of `--immersive`, starts in full-view stereo: each
    /// game frame is replayed once per eye and fills that eye's view,
    /// instead of one image on the virtual screen. Stage A keeps the game's
    /// camera as the head, so the world turns with the head until the
    /// game's view follows the head pose (stage B). Both thumbstick clicks
    /// together, or F8 on a keyboard, flip between stereo and the screen
    /// (StereoToggle). Like `--immersive`, the runtime never sees it.
    nonisolated static let stereoFlag = "--stereo"
    nonisolated static let wantsStereo = isImmersive
        && (ProcessInfo.processInfo.arguments.dropFirst().contains(stereoFlag) || wantsFullView)

    /// The launcher whose runtime is running, for the runtime's C callback,
    /// which cannot capture a context.
    private static weak var active: GameLauncher?

    private(set) var state = State.idle
    let gameFolder = URL.documentsDirectory.appending(path: "VTMB", directoryHint: .isDirectory)
    let overlayFolder = URL.applicationSupportDirectory.appending(path: "overlay", directoryHint: .isDirectory)

    var logFolder: URL {
        overlayFolder.appending(path: "logs", directoryHint: .isDirectory)
    }

    /// Starts the runtime once, if the game data is in place. Calling it
    /// again while the runtime runs does nothing; the runtime cannot be
    /// started twice in one process.
    func startIfReady() {
        guard state == .idle || state == .missingData else { return }
        let fileManager = FileManager.default
        // Made up front so the Files app shows the folder to copy into.
        try? fileManager.createDirectory(at: gameFolder, withIntermediateDirectories: true)
        guard hasGameData(fileManager: fileManager) else {
            state = .missingData
            return
        }
        do {
            try fileManager.createDirectory(at: overlayFolder, withIntermediateDirectories: true)
        } catch {
            state = .failedToStart(Int32(EIO))
            return
        }
        // Sound and movies are on for play; benchmark launches add --nosound
        // and --novid themselves so their numbers match the Mac's.
        var baseArguments = [
            "xrhost", "--run", "--gfx", "metal", "--guest-quiet",
            "--game", gameFolder.path(percentEncoded: false),
            "--overlay", overlayFolder.path(percentEncoded: false),
        ]
        if Self.wantsFullView {
            baseArguments += ["--aspect", "\(Self.desktopSize.width)x\(Self.desktopSize.height)"]
        }
        // Set here, on the main thread, before the runtime's thread exists,
        // because setenv is not safe against a concurrent getenv.
        Self.resolveTracePaths(in: overlayFolder)
        // The game only plays audio. Asking SDL for the playback category
        // keeps it from opening a session that touches the microphone.
        setenv("SDL_AUDIO_CATEGORY", "playback", 0)
        // Before the device exists, so its first Present already goes to
        // the compositor's ring, and to the window's layer as well until
        // the space has shown the game (WindowCoordinator).
        if Self.isImmersive {
            metal_d3d9_space_request()
            if Self.wantsStereo {
                metal_d3d9_stereo_enable(1)
            }
            if Self.wantsFullView {
                metal_d3d9_fullview_enable(1)
            }
        }
        var arguments = baseArguments + Self.launchArguments(resolvingPathsIn: overlayFolder)
        // camera_prefs is a bit per weapon class kept in first person:
        // ranged 2, thrown 4. A launch here once saved 0, which sends guns
        // to third person, so full view sets the game's default, 6, back.
        // Melee goes to third person whatever it says; the runtime reads
        // that from the game and holds the head for it.
        if Self.wantsFullView {
            if !arguments.contains("--") { arguments.append("--") }
            arguments += ["+camera_prefs", "6"]
            // With hand aim the reticle is drawn at the gun hand, so the
            // game's own, fixed in the middle of the head's view, goes.
            if SpatialInput.handAim { arguments += ["+crosshair", "0"] }
        }
        Self.active = self
        let result = Self.startRuntime(arguments, desktopWidth: Self.desktopSize.width, desktopHeight: Self.desktopSize.height)
        state = result == 0 ? .running : .failedToStart(result)
        // From the start, not from the space opening: a controller can
        // connect before the space exists, and in the window as well.
        if result == 0 {
            ControllerWatcher.shared.start()
            Self.watchBackground()
        }
    }

    /// Taking the headset off, or losing both the space and the window,
    /// sends the app to the background, where the system holds its GPU
    /// work and a Present waits until it returns. The runtime is told, so
    /// its graphics watchdog does not end the game over that wait.
    private static func watchBackground() {
        let center = NotificationCenter.default
        center.addObserver(forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: .main) { _ in
            rt_host_background(1)
        }
        center.addObserver(forName: UIApplication.willEnterForegroundNotification, object: nil, queue: .main) { _ in
            rt_host_background(0)
        }
    }

    private func runtimeEnded(status: Int32, reason: String?) {
        state = .stopped(status: status, reason: reason)
    }

    /// The runtime opens the game executable by name before its own
    /// case-insensitive file layer is up, so this only asks whether it is
    /// there in any case; the runtime decides whether the rest is usable.
    private func hasGameData(fileManager: FileManager) -> Bool {
        let names = (try? fileManager.contentsOfDirectory(atPath: gameFolder.path(percentEncoded: false))) ?? []
        return names.contains { $0.lowercased() == "vampire.exe" }
    }

    /// The arguments the app itself was launched with (from `devicectl device
    /// process launch`, after the bundle ID), for benchmark runs: runtime
    /// options such as `--input-script` and `--close-after`, then `--` and
    /// the game's own arguments. A normal launch has none, so the runtime
    /// gets exactly the base arguments.
    ///
    /// They follow the base arguments unchanged because the runtime takes its
    /// options in any order and the base holds no `--`. A relative input
    /// script path is made absolute inside the overlay, since the runtime
    /// reads the script while it parses its options and the app's working
    /// directory is the root of the file system, outside the container.
    ///
    /// devicectl's own parser may take a `--` in its arguments as the end of
    /// its options and drop it, so a game argument such as `+load` reaching
    /// here before any `--` gets one put back in front of it. Game console
    /// commands start with `+` and runtime options never do.
    private static func launchArguments(resolvingPathsIn overlay: URL) -> [String] {
        var arguments = Array(ProcessInfo.processInfo.arguments.dropFirst())
        for hostFlag in [immersiveFlag, stereoFlag, fullViewFlag, noFullViewFlag] {
            if let flag = arguments.firstIndex(of: hostFlag), arguments[..<flag].allSatisfy({ $0 != "--" }) {
                arguments.remove(at: flag)
            }
        }
        var index = 0
        while index < arguments.count, arguments[index] != "--" {
            if arguments[index].hasPrefix("+") {
                arguments.insert("--", at: index)
                break
            }
            if arguments[index] == "--input-script", index + 1 < arguments.count {
                arguments[index + 1] = path(arguments[index + 1], resolvedIn: overlay)
                index += 1
            }
            index += 1
        }
        return arguments
    }

    /// The runtime reads its `RT_*` settings with getenv in this same
    /// process, so the environment the launch gave the app already reaches
    /// it; only the trace files the runtime writes by path need a relative
    /// path made absolute inside the overlay, for the same reason as the
    /// input script.
    private static func resolveTracePaths(in overlay: URL) {
        let environment = ProcessInfo.processInfo.environment
        for name in ["RT_FRAME_TRACE", "RT_PRESENT_TRACE", "RT_COMPOSITOR_TRACE"] {
            guard let value = environment[name], !value.isEmpty, !value.hasPrefix("/") else { continue }
            let resolved = path(value, resolvedIn: overlay)
            // The runtime opens the trace with fopen, which makes no folders.
            try? FileManager.default.createDirectory(
                at: URL(filePath: resolved).deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            setenv(name, resolved, 1)
        }
    }

    private static func path(_ path: String, resolvedIn folder: URL) -> String {
        guard !path.hasPrefix("/") else { return path }
        return folder.appending(path: path, directoryHint: .notDirectory).path(percentEncoded: false)
    }

    /// rt_host_start copies argv before it returns, so the C strings are
    /// freed here.
    ///
    /// Nonisolated, so the callback written inside it is too. Written in a
    /// main actor method, the closure would be inferred main actor isolated,
    /// and Swift 6 checks that isolation when the runtime calls it on its
    /// own thread, which traps.
    nonisolated private static func startRuntime(_ arguments: [String], desktopWidth: Int, desktopHeight: Int) -> Int32 {
        let strings = arguments.map { strdup($0) }
        defer { strings.forEach { free($0) } }
        let argv = strings.map { UnsafePointer($0) }
        return argv.withUnsafeBufferPointer { buffer in
            rt_host_start(
                Int32(buffer.count), buffer.baseAddress,
                Int32(desktopWidth), Int32(desktopHeight)
            ) { status, reason in
                // On the runtime's thread, after a refused or returned run.
                // The reason is only valid during this call, so it is copied
                // before the hop to the main actor, which owns the state.
                let copiedReason = reason.map { String(cString: $0) }
                Task { @MainActor in
                    GameLauncher.active?.runtimeEnded(status: status, reason: copiedReason)
                }
            }
        }
    }
}
