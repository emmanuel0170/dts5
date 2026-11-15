import 'dart:math' as math;

import '../engine.dart';
import '../planning/planner.dart';
import 'guidance.dart';
import 'phrase_bank.dart';

/// One continuous sound stream, as the native synth renders it.
class Cue {
  const Cue({required this.pan, required this.ppm, required this.hz, required this.gain, this.locked = false});

  final double pan; // -1 left .. +1 right (needs stereo headphones to mean anything)
  final double ppm; // pulses per minute; 0 = continuous tone
  final double hz;
  final double gain; // 0..1
  final bool locked; // beacon only: double-pulse accent when the way ahead is straight

  Map<String, Object> toMap() => {'pan': pan, 'ppm': ppm, 'hz': hz, 'gain': gain, 'locked': locked};

  @override
  String toString() => 'Cue(pan ${pan.toStringAsFixed(2)}, ${ppm.toStringAsFixed(0)} ppm, ${hz.toStringAsFixed(0)} Hz, '
      'gain ${gain.toStringAsFixed(2)}${locked ? ', locked' : ''})';
}

class CueFrame {
  const CueFrame({this.hazards = const [], this.beacon, this.earcons = const []});

  /// [left, centre, right]: the nearest obstacle in each lateral band, each its own tone, panned to where it is.
  /// null = silent.
  final List<Cue?> hazards;
  final Cue? beacon; // the way to walk toward the search target
  final List<String> earcons; // one-shots: found, arrived, lost, alert

  bool get anyHazard => hazards.any((h) => h != null);

  Map<String, Object?> toMap() => {
        'hazards': [for (final h in hazards) h?.toMap()],
        'beacon': beacon?.toMap(),
        'earcons': earcons,
      };
}

/// Starting values, not tuned ones. They follow published obstacle-sonification work (distance -> pulse rate,
/// position -> stereo pan, roughly 50-280 pulses/min) but must be tested with real users.
class CueConfig {
  const CueConfig({
    this.beaconNearM = 0.9,
    this.beaconFarM = 5.0,
    this.beaconNearPpm = 180,
    this.beaconFarPpm = 30,
    this.beaconUnknownPpm = 60,
    this.beaconHz = 880,
    this.beaconGain = 0.45,
    this.lockedDeg = 7,
    this.hazardNearM = 0.5,
    this.hazardFarM = 2.0,
    this.hazardNearPpm = 280,
    this.hazardFarPpm = 60,
    this.hazardContinuousM = 0.55,
    this.hazardHz = const [300.0, 340.0, 380.0], // slightly different pitches so simultaneous tones stay separable
    this.hazardGainNear = 0.9,
    this.hazardGainFar = 0.45,
    this.hazardPanDeg = 25, // an obstacle this far off-axis is panned fully to that side
    this.beaconGainWhileHazard = 0.5,
    this.debounce = 2, // consecutive depth updates a sector must stay occupied before it sounds
    this.planMaxAgeMs = 1500,
  });

  final double beaconNearM, beaconFarM, beaconNearPpm, beaconFarPpm, beaconUnknownPpm;
  final double beaconHz, beaconGain, lockedDeg;
  final double hazardNearM, hazardFarM, hazardNearPpm, hazardFarPpm, hazardContinuousM;
  final List<double> hazardHz;
  final double hazardGainNear, hazardGainFar, hazardPanDeg, beaconGainWhileHazard;
  final int debounce;
  final double planMaxAgeMs;
}

/// Speech says WHAT and WHEN (found, lost, arrived, which hazard). Sound says WHERE and HOW CLOSE, continuously:
///  - up to three hazard tones, one per lateral band, each panned to the obstacle's own bearing, so an obstacle on
///    the left is heard on the left and one on the right on the right;
///  - a beacon that pans toward the way to walk (the planned path around obstacles, not just the straight line to
///    the target).
class CueMapper {
  CueMapper([this.cfg = const CueConfig()]);

  final CueConfig cfg;
  final List<int> _streak = [0, 0, 0];
  double? _lastStamp;

  /// Call right after [GuidanceEngine.update]; [spoken] is what that call returned. [plan] is the latest path plan
  /// (from the depth grid); without a usable plan the coarse 3x5 depth grid in [frame] is the fallback.
  CueFrame map({
    required GuidanceEngine g,
    required List<Utterance> spoken,
    FusedFrame? frame,
    PlanResult? plan,
    double nowMs = 0,
  }) {
    final earcons = <String>[];
    for (final u in spoken) {
      final k = u.parts.isEmpty ? '' : u.parts.first;
      if (k == 'say:found') {
        earcons.add('found');
      } else if (k == 'say:arrived') {
        earcons.add('arrived');
      } else if (k.startsWith('lost:')) {
        earcons.add('lost');
      } else if (k == 'say:careful') {
        earcons.add('alert');
      }
    }

    // ---- nearest obstacle per lateral band ----
    final ranges = <double?>[null, null, null];
    final azimuth = <double>[0, 0, 0];
    double? stamp;
    final usable = plan != null && plan.ok && nowMs - plan.tMs <= cfg.planMaxAgeMs;
    if (usable) {
      stamp = plan.tMs;
      final hits = [plan.left, plan.center, plan.right];
      double? goalRange, goalAz;
      if (plan.hasGoal && plan.goalX != null && plan.goalZ != null) {
        goalRange = math.sqrt(plan.goalX! * plan.goalX! + plan.goalZ! * plan.goalZ!);
        goalAz = math.atan2(plan.goalX!, plan.goalZ!) * 180 / math.pi;
      }
      final nearTarget = g.phase == Phase.guiding || g.phase == Phase.arrived;
      for (var k = 0; k < 3; k++) {
        final h = hits[k];
        if (h == null || h.rangeM > cfg.hazardFarM) continue;
        // The thing we are walking toward is an obstacle too, but not a hazard.
        if (nearTarget && goalRange != null && (h.azimuthDeg - goalAz!).abs() < 20 && h.rangeM > goalRange - 0.9) {
          continue;
        }
        ranges[k] = h.rangeM;
        azimuth[k] = h.azimuthDeg;
      }
    } else if (frame != null && frame.grid != null && frame.gridRows >= 2 && frame.gridCols >= 3) {
      stamp = frame.tMs;
      _fallbackBands(frame, g, ranges, azimuth);
    }
    if (stamp != null && stamp != _lastStamp) {
      _lastStamp = stamp;
      for (var k = 0; k < 3; k++) {
        _streak[k] = ranges[k] != null ? _streak[k] + 1 : 0;
      }
    }

    final hazards = <Cue?>[null, null, null];
    for (var k = 0; k < 3; k++) {
      final r = ranges[k];
      if (r == null || _streak[k] < cfg.debounce) continue;
      final t = _t(r, cfg.hazardNearM, cfg.hazardFarM);
      hazards[k] = Cue(
        pan: (azimuth[k] / cfg.hazardPanDeg).clamp(-1.0, 1.0).toDouble(),
        ppm: r <= cfg.hazardContinuousM ? 0 : _lerp(cfg.hazardNearPpm, cfg.hazardFarPpm, t),
        hz: cfg.hazardHz[k],
        gain: _lerp(cfg.hazardGainNear, cfg.hazardGainFar, t),
      );
    }
    final anyHazard = hazards.any((h) => h != null);

    // ---- beacon ----
    Cue? beacon;
    final target = g.activeTarget;
    if (g.phase == Phase.guiding && target != null) {
      var bearing = g.camera.bearingDeg(target.cx);
      if (usable && plan.hasGoal && plan.steerDeg != null) bearing = plan.steerDeg!; // go around, not through
      final half = g.camera.hfovDeg / 2;
      final d = target.depthMedian;
      final known = d != null && !d.isNaN;
      beacon = Cue(
        pan: (bearing / half).clamp(-1.0, 1.0).toDouble(),
        ppm: known
            ? _lerp(cfg.beaconNearPpm, cfg.beaconFarPpm, _t(d, cfg.beaconNearM, cfg.beaconFarM))
            : cfg.beaconUnknownPpm,
        hz: cfg.beaconHz,
        gain: cfg.beaconGain * (anyHazard ? cfg.beaconGainWhileHazard : 1.0),
        locked: bearing.abs() < cfg.lockedDeg,
      );
    }
    return CueFrame(hazards: hazards, beacon: beacon, earcons: earcons);
  }

  /// No usable plan (not enough floor in view, or no depth yet): read the middle row of the coarse depth grid.
  void _fallbackBands(FusedFrame f, GuidanceEngine g, List<double?> ranges, List<double> azimuth) {
    final grid = f.grid!;
    final row = f.gridRows ~/ 2;
    final cols = f.gridCols;
    final bands = <List<int>>[
      [for (var c = 0; c < cols ~/ 2; c++) c], // left: columns left of centre
      [cols ~/ 2], // centre
      [for (var c = cols ~/ 2 + 1; c < cols; c++) c], // right
    ];
    for (var k = 0; k < 3; k++) {
      double? best;
      var bestCol = bands[k].isEmpty ? cols ~/ 2 : bands[k].first;
      for (final c in bands[k]) {
        final v = grid[row * cols + c];
        if (v.isNaN) continue;
        if (best == null || v < best) {
          best = v;
          bestCol = c;
        }
      }
      if (best != null && best <= cfg.hazardFarM) {
        ranges[k] = best;
        azimuth[k] = g.camera.bearingDeg((bestCol + 0.5) / cols);
      }
    }
  }

  static double _t(double v, double a, double b) => ((v - a) / (b - a)).clamp(0.0, 1.0).toDouble();
  static double _lerp(double a, double b, double t) => a + (b - a) * t;
}
