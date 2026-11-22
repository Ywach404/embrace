import SwiftUI
import UIKit

/// Opens the immersive space, keeps the app's window out of the way while
/// the game is in it, and brings the window back when the space closes.
///
/// On visionOS a game controller's input reaches the app only while it has
/// an immersive space open, and looking at one of the app's windows hands
/// the controller back to the system until the gaze moves away. So once the
/// space has shown its first game frame the window is hidden and its scene
/// dismissed; the immersive space alone keeps the app running, the game
/// included.
///
/// The order lives in the backend (`metal_d3d9_space_*`, space.h), where a
/// Mac test drives it: the window keeps getting frames until the space has
/// shown one, frames go to the ring from the request, and the window's
/// layer is off before the window goes. This class only acts on its
/// answers.
///
/// The open is driven by a task this class owns, not by a view's task.
/// SwiftUI cancels and restarts a view's `.task(id:)` when the id changes,
/// and the launcher's state changes from idle to running in the same
/// view's `onAppear`: the first task ran, took a one-shot "opening" flag,
/// was cancelled while waiting for the first frame, and the restarted task
/// found the flag taken. The space never opened, and nothing was logged.
///
/// SDL never moves its UIWindow to another scene. When the space closes
/// after the window went, a new window scene is opened and the game's
/// UIWindow (from the runtime, `rt_host_game_window`) is moved into it
/// before frames go back to the window and input back to SDL's window
/// focus. Until then the host's input claim stays on, so the moment between
/// the space closing and the window returning, when the app may be briefly
/// inactive, does not reach the game as a focus loss that would mute it.
///
/// Once the window has the game back, its Return to Immersive button
/// (`returnToImmersive`) asks for the space again and runs the same order
/// as the first open, as many times as it is tapped. The tap only starts
/// the driver; like the first open's, the task is this class's, so no
/// view's lifetime can cancel it midway with the claim taken.
@MainActor
@Observable
final class WindowCoordinator {
    static let shared = WindowCoordinator()

    static let windowID = "game"

    /// How often the driver asks whether the space is due, and checks the
    /// watchdog, until it has opened or is not wanted.
    private static let pollInterval = Duration.milliseconds(100)

    /// The window has the game back after a space closed, so the space can
    /// be opened again: what shows the window's Return to Immersive button.
    private(set) var canReturnToImmersive = false

    /// A new window scene was asked for after the space closed, to take the
    /// game's window back.
    @ObservationIgnored private var returning = false
    @ObservationIgnored private var driver: Task<Void, Never>?
    @ObservationIgnored private var openWindow: OpenWindowAction?
    @ObservationIgnored private var dismissWindow: DismissWindowAction?
    @ObservationIgnored private var openImmersiveSpace: OpenImmersiveSpaceAction?

    /// The scene actions, from the view currently in the window scene. They
    /// act on the app's scenes by ID, not on the view's own scene, so a view
    /// that has gone leaves them usable.
    func remember(
        openWindow: OpenWindowAction, dismissWindow: DismissWindowAction, openImmersiveSpace: OpenImmersiveSpaceAction
    ) {
        self.openWindow = openWindow
        self.dismissWindow = dismissWindow
        self.openImmersiveSpace = openImmersiveSpace
    }

    /// Starts the driver once the runtime runs. Calling it again, from a
    /// view that appeared again or a state that changed, adds no second
    /// driver; and if one ever did run, the backend gives the open to one
    /// of them only.
    func runtimeStateChanged(running: Bool) {
        guard GameLauncher.isImmersive, running else { return }
        startDriver()
    }

    /// Return to Immersive. The run's own mode comes back with it: a
    /// `--stereo` run reopens in stereo even if the toggle left it on the
    /// screen. The backend takes the request only from the window, so a
    /// second tap before the space opens does nothing.
    func returnToImmersive() {
        guard GameLauncher.isImmersive, canReturnToImmersive else { return }
        metal_d3d9_stereo_enable(GameLauncher.wantsStereo ? 1 : 0)
        guard metal_d3d9_space_reopen() != 0 else {
            ScreenCompositor.log("Return to Immersive: the space is not ready to open again")
            return
        }
        canReturnToImmersive = false
        ScreenCompositor.log(
            "Return to Immersive: opening the space again"
                + (GameLauncher.wantsStereo ? " in full-view stereo" : "")
        )
        startDriver()
    }

    /// One driver at a time, cleared when it finishes so the next open
    /// gets its own. A driver that finds nothing to open returns at once.
    private func startDriver() {
        guard driver == nil else { return }
        driver = Task { [weak self] in
            await self?.driveOpen()
            self?.driver = nil
        }
    }

    /// Polls until the backend hands over the open (a game frame is in the
    /// ring, so SDL's window has taken the window scene and the space's
    /// scene cannot be picked up as the window's), opens the space, and
    /// reports the result. Nothing cancels this task, and a sleep cut
    /// short only polls again sooner.
    private func driveOpen() async {
        while true {
            metal_d3d9_space_watchdog()
            switch metal_d3d9_space_should_open() {
            case 1:
                if await open() { return }
            case 0:
                try? await Task.sleep(for: Self.pollInterval)
            default:
                return
            }
        }
    }

    /// Returns false when the open was handed back, to be tried again.
    private func open() async -> Bool {
        guard let openImmersiveSpace else {
            metal_d3d9_space_open_result(Int32(METAL_D3D9_SPACE_ABANDONED))
            try? await Task.sleep(for: Self.pollInterval)
            return false
        }
        switch await openImmersiveSpace(id: GameLauncher.immersiveSpaceID) {
        case .opened:
            rt_host_input_claim(1)
            MouseForwarder.shared.start()
            StereoToggle.shared.start()
            metal_d3d9_space_open_result(Int32(METAL_D3D9_SPACE_OPENED))
            ScreenCompositor.log("immersive space opened; the host owns input")
            // The compositor may have shown its first frame before this
            // answer came back.
            dismissWindowIfShown()
        case .userCancelled:
            metal_d3d9_space_open_result(Int32(METAL_D3D9_SPACE_DECLINED))
            ScreenCompositor.log("immersive space declined; showing the game in the window")
            canReturnToImmersive = true
        case .error:
            metal_d3d9_space_open_result(Int32(METAL_D3D9_SPACE_FAILED))
            ScreenCompositor.log("immersive space failed to open; showing the game in the window")
            canReturnToImmersive = true
        @unknown default:
            metal_d3d9_space_open_result(Int32(METAL_D3D9_SPACE_FAILED))
            ScreenCompositor.log("immersive space not opened; showing the game in the window")
            canReturnToImmersive = true
        }
        return true
    }

    /// From the compositor, once a frame with the game on the screen has
    /// completed on its GPU (already recorded in the backend).
    func screenShowedFirstFrame() {
        dismissWindowIfShown()
    }

    /// Dismisses the window once the space is open and has shown a game
    /// frame. The backend has turned the window's layer off by the time it
    /// says so, so the render thread never waits on a layer with no scene.
    private func dismissWindowIfShown() {
        guard let dismissWindow, metal_d3d9_space_take_dismiss() != 0 else { return }
        // Hidden first, so nothing of the game's window is drawn while the
        // scene goes away; SDL's own state does not follow UIKit's here.
        gameWindow()?.isHidden = true
        dismissWindow(id: Self.windowID)
        ScreenCompositor.log("window dismissed; only the immersive space is shown")
    }

    /// The space closed (the Digital Crown, or the app leaving it).
    func spaceClosed() {
        guard metal_d3d9_space_closed() != 0 else {
            // The window never went: the frames are back on it already.
            rt_host_input_claim(0)
            canReturnToImmersive = true
            ScreenCompositor.log("immersive space closed; back to the window")
            return
        }
        guard let openWindow else {
            returnToWindow()
            return
        }
        returning = true
        openWindow(id: Self.windowID)
        ScreenCompositor.log("immersive space closed; opening the window again")
    }

    /// A view in a window scene reports the scene it is in. While
    /// returning, the game's window moves into it and the game goes back
    /// to the window.
    func windowSceneAppeared(_ scene: UIWindowScene) {
        guard returning else { return }
        returning = false
        if let window = gameWindow(), window.windowScene !== scene {
            window.windowScene = scene
            window.isHidden = false
            window.makeKeyAndVisible()
            ScreenCompositor.log("game window moved into the new window scene")
        }
        returnToWindow()
    }

    private func returnToWindow() {
        metal_d3d9_space_window_back()
        rt_host_input_claim(0)
        canReturnToImmersive = true
        ScreenCompositor.log("back to the window")
    }

    private func gameWindow() -> UIWindow? {
        guard let pointer = rt_host_game_window() else { return nil }
        return Unmanaged<UIWindow>.fromOpaque(pointer).takeUnretainedValue()
    }
}

/// Reports the window scene its view lands in, so the coordinator can move
/// the game's window there. The scene is known only once the view is in a
/// window, which SwiftUI does not expose by itself.
struct WindowSceneReader: UIViewRepresentable {
    let onScene: @MainActor (UIWindowScene) -> Void

    func makeUIView(context: Context) -> SceneReportingView {
        let view = SceneReportingView()
        view.onScene = onScene
        view.isUserInteractionEnabled = false
        return view
    }

    func updateUIView(_ uiView: SceneReportingView, context: Context) {
        uiView.onScene = onScene
    }

    final class SceneReportingView: UIView {
        var onScene: (@MainActor (UIWindowScene) -> Void)?

        override func didMoveToWindow() {
            super.didMoveToWindow()
            if let scene = window?.windowScene {
                onScene?(scene)
            }
        }
    }
}
