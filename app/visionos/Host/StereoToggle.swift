import Foundation
import GameController

/// Flips the immersive view between full-view stereo and the virtual screen:
/// both thumbstick clicks pressed together on the pad, or F8 on a paired
/// keyboard.
///
/// The chord is one the game has no use for, and it is read here as well as
/// by the game: SDL sees both clicks too, which the pad layer takes as two
/// stick clicks, so the chord is meant to be pressed and released quickly.
/// The flip itself is the backend's (`metal_d3d9_stereo_enable`), which takes
/// effect at the next game frame's start; the compositor follows the frames
/// it is handed, two layers or one, so the mode and the picture can never
/// disagree for a frame.
///
/// Polled on the main queue at 20 Hz rather than through value handlers,
/// since SDL owns the controllers' handlers and setting them here would take
/// them away from it.
final class StereoToggle: @unchecked Sendable {
    nonisolated static let shared = StereoToggle()

    private static let pollInterval = DispatchTimeInterval.milliseconds(50)

    private var timer: DispatchSourceTimer?
    private var chordWasDown = false
    private var keyWasDown = false

    /// Called on the main thread once the space is open.
    nonisolated func start() {
        guard timer == nil else { return }
        let timer = DispatchSource.makeTimerSource(queue: .main)
        timer.schedule(deadline: .now() + Self.pollInterval, repeating: Self.pollInterval)
        timer.setEventHandler { [self] in
            self.poll()
        }
        timer.resume()
        self.timer = timer
        ScreenCompositor.log(
            "stereo toggle ready (both stick clicks or F8); starting in "
                + (metal_d3d9_stereo_enabled() != 0 ? "full-view stereo" : "the virtual screen")
        )
    }

    private func poll() {
        var chord = false
        for controller in GCController.controllers() {
            guard let pad = controller.extendedGamepad,
                let left = pad.leftThumbstickButton,
                let right = pad.rightThumbstickButton
            else { continue }
            if left.isPressed && right.isPressed { chord = true }
        }
        let key = GCKeyboard.coalesced?.keyboardInput?.button(forKeyCode: .F8)?.isPressed ?? false
        // On the press, not while held, so holding the chord flips once.
        if (chord && !chordWasDown) || (key && !keyWasDown) {
            let on = metal_d3d9_stereo_enabled() == 0
            metal_d3d9_stereo_enable(on ? 1 : 0)
            ScreenCompositor.log("stereo toggle: " + (on ? "full-view stereo" : "the virtual screen"))
        }
        chordWasDown = chord
        keyWasDown = key
    }
}
