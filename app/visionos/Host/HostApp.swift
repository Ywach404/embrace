import CompositorServices
import SwiftUI

/// One window. Its size is the runtime's pinned desktop, so the 4:3 game
/// window the runtime fits inside it fills the window exactly.
///
/// With `--immersive` the game is shown in an immersive space instead, on a
/// virtual screen drawn through CompositorServices (ScreenCompositor). Once
/// the screen shows the game, the window is dismissed so none sits under
/// the user's gaze, and it comes back when the space closes
/// (WindowCoordinator). The game never sees the window go: SDL's window
/// lives on without a scene, and closing it is not taken for a quit.
@main
struct HostApp: App {
    @State private var launcher = GameLauncher()
    /// Full view is the world around the player: full immersion, so a held
    /// view (dialogue, third person) shows black past the game's view
    /// rather than the room, which read as a window onto the world
    /// (headset run, 2026-11-21). Progressive immersion would keep the
    /// Home button, but CompositorServices stopped the app at the first
    /// drawable presented in it (cp_drawable_encode_present, three runs).
    /// The theater screen keeps the room.
    @State private var immersion: ImmersionStyle =
        GameLauncher.wantsFullView ? FullImmersionStyle.full as any ImmersionStyle : MixedImmersionStyle.mixed

    var body: some Scene {
        WindowGroup(id: WindowCoordinator.windowID) {
            ContentView(launcher: launcher)
        }
        .defaultSize(width: CGFloat(GameLauncher.desktopSize.width), height: CGFloat(GameLauncher.desktopSize.height))

        ImmersiveSpace(id: GameLauncher.immersiveSpaceID) {
            CompositorLayer(configuration: ScreenLayerConfiguration()) { renderer in
                ScreenCompositor.start(renderer) {
                    // The space was closed (the Digital Crown, or the app
                    // leaving it): the window comes back, then frames go
                    // back to it and input back to SDL's window focus.
                    Task { @MainActor in
                        WindowCoordinator.shared.spaceClosed()
                    }
                }
            }
        }
        .immersionStyle(selection: $immersion, in: .mixed, .full)
        // In full view the player's own hands and the controllers would show
        // through the game, over the gun, the reticle and the gauge cards at
        // the hands (headset run, 2026-11-21); the game draws what is held.
        .upperLimbVisibility(GameLauncher.wantsFullView ? .hidden : .automatic)
    }
}
