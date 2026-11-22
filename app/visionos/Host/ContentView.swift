import SwiftUI

/// What the window shows before SDL's own window takes the scene over: where
/// the game data goes when it is missing, or why the runtime stopped.
struct ContentView: View {
    let launcher: GameLauncher
    @Environment(\.openImmersiveSpace) private var openImmersiveSpace
    @Environment(\.openWindow) private var openWindow
    @Environment(\.dismissWindow) private var dismissWindow

    var body: some View {
        Group {
            switch launcher.state {
            case .idle, .running:
                ProgressView("Starting")
            case .missingData:
                MissingDataView(folder: launcher.gameFolder) {
                    launcher.startIfReady()
                }
            case .stopped(let status, let reason):
                StoppedView(
                    title: "The runtime stopped (status \(status)).",
                    reason: reason,
                    detail: "Its log is in \(launcher.logFolder.path(percentEncoded: false))."
                )
            case .failedToStart(let error):
                StoppedView(
                    title: "The runtime thread could not be started (error \(error)).",
                    detail: "Nothing was run."
                )
            }
        }
        .padding(40)
        .background {
            // The scene this window landed in: a window opened again after
            // the space closed takes the game's window back here.
            WindowSceneReader { scene in
                WindowCoordinator.shared.windowSceneAppeared(scene)
            }
        }
        .onAppear {
            WindowCoordinator.shared.remember(
                openWindow: openWindow, dismissWindow: dismissWindow, openImmersiveSpace: openImmersiveSpace
            )
            launcher.startIfReady()
            WindowCoordinator.shared.runtimeStateChanged(running: launcher.state == .running)
        }
        .onChange(of: launcher.state) { _, state in
            WindowCoordinator.shared.runtimeStateChanged(running: state == .running)
        }
        // An ornament, outside the window's bounds: once the game is back
        // in this window, SDL's own UIWindow covers everything inside them.
        .ornament(
            visibility: WindowCoordinator.shared.canReturnToImmersive ? .visible : .hidden,
            attachmentAnchor: .scene(.bottom),
            contentAlignment: .top
        ) {
            ReturnToImmersiveButton()
        }
    }
}

/// Opens the immersive space again after it closed (the Digital Crown).
/// The action only asks the coordinator, which owns the open from there,
/// so nothing about this view can cancel it.
private struct ReturnToImmersiveButton: View {
    var body: some View {
        Button {
            WindowCoordinator.shared.returnToImmersive()
        } label: {
            Label("Return to Immersive", systemImage: "visionpro")
                .font(.title3)
                .padding(.horizontal, 8)
        }
        .padding(12)
        .glassBackgroundEffect()
    }
}

private struct MissingDataView: View {
    let folder: URL
    let checkAgain: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("No game data found.")
                .font(.title)
            Text(
                """
                This app carries no game files. Copy the contents of your own installed copy of the game \
                into the VTMB folder this app shares in the Files app (On My Apple Vision Pro, then this \
                app, then VTMB), so that Vampire.exe sits directly inside VTMB.
                """
            )
            Text(folder.path(percentEncoded: false))
                .font(.footnote.monospaced())
                .textSelection(.enabled)
            Button("Check Again", action: checkAgain)
        }
        .frame(maxWidth: 720, alignment: .leading)
    }
}

private struct StoppedView: View {
    let title: String
    /// The runtime's own words for a refused run, which otherwise reach only
    /// stderr and, once its log folder exists, the run log.
    var reason: String?
    let detail: String

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(title)
                .font(.title)
            if let reason {
                Text(reason)
                    .font(.body.monospaced())
                    .textSelection(.enabled)
            }
            Text(detail)
                .font(.footnote.monospaced())
                .textSelection(.enabled)
        }
        .frame(maxWidth: 720, alignment: .leading)
    }
}
