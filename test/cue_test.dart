import 'package:flutter_test/flutter_test.dart';
import 'package:dts5/decision/cue_mapper.dart';
import 'package:dts5/decision/guidance.dart';
import 'package:dts5/decision/phrase_bank.dart';
import 'package:dts5/decision/scene_tracker.dart';
import 'package:dts5/planning/planner.dart';

import 'helpers.dart';

PlanResult plan(
  double t, {
  SectorHit? left,
  SectorHit? center,
  SectorHit? right,
  bool hasGoal = false,
  double? steer,
  double gx = 0,
  double gz = 4,
}) =>
    PlanResult(
      status: PlanStatus.ok,
      tMs: t,
      left: left,
      center: center,
      right: right,
      steerDeg: steer,
      hasGoal: hasGoal,
      goalX: gx,
      goalZ: gz,
    );

void main() {
  late SceneTracker tr;
  late GuidanceEngine g;
  late CueMapper mapper;
  var t = 1000.0;

  List<Utterance> step(List<DetectionObj> dets, {List<double>? grid}) {
    final f = frame(t, dets: dets, grid: grid);
    t += 300;
    tr.update(f);
    return g.update(f, tr);
  }

  CueFrame stepCues(List<DetectionObj> dets, {List<double>? grid}) {
    final spoken = step(dets, grid: grid);
    return mapper.map(g: g, spoken: spoken);
  }

  /// One frame at time [at], mapped with the given plan.
  CueFrame mapAt(double at, {List<DetectionObj> dets = const [], List<double>? grid, PlanResult? p}) {
    final f = frame(at, dets: dets, grid: grid);
    tr.update(f);
    final spoken = g.update(f, tr);
    return mapper.map(g: g, spoken: spoken, frame: f, plan: p, nowMs: at);
  }

  setUp(() {
    tr = SceneTracker();
    g = GuidanceEngine();
    mapper = CueMapper();
    t = 1000.0;
  });

  group('beacon', () {
    test('appears once the target is found, pans toward it, with a found earcon', () {
      g.setGoal(SearchGoal(labels: ['chair'], phrase: 'a chair'));
      expect(stepCues([det('chair', cx: 0.3, depth: 3.0)]).beacon, isNull); // not confirmed yet
      final c = stepCues([det('chair', cx: 0.3, depth: 3.0)]);
      expect(c.earcons, ['found']);
      expect(c.beacon, isNotNull);
      expect(c.beacon!.pan, lessThan(0)); // target is on the left
      expect(c.beacon!.pan, greaterThan(-1));
      expect(c.beacon!.locked, isFalse);
    });

    test('pulse rate rises as the target gets closer', () {
      g.setGoal(SearchGoal(labels: ['chair'], phrase: 'a chair'));
      stepCues([det('chair', cx: 0.5, depth: 4.0)]);
      final far = stepCues([det('chair', cx: 0.5, depth: 4.0)]).beacon!;
      CueFrame c = const CueFrame();
      for (var i = 0; i < 12; i++) {
        c = stepCues([det('chair', cx: 0.5, depth: 1.5)]);
      }
      expect(c.beacon, isNotNull);
      expect(c.beacon!.ppm, greaterThan(far.ppm));
    });

    test('locked accent when the way ahead is straight', () {
      g.setGoal(SearchGoal(labels: ['chair'], phrase: 'a chair'));
      stepCues([det('chair', cx: 0.5, depth: 3.0)]);
      final c = stepCues([det('chair', cx: 0.5, depth: 3.0)]);
      expect(c.beacon!.locked, isTrue);
      expect(c.beacon!.pan, closeTo(0, 1e-6));
    });

    test('beacon stops and an arrived earcon plays on arrival', () {
      g.setGoal(SearchGoal(labels: ['chair'], phrase: 'a chair'));
      final all = <CueFrame>[];
      for (var i = 0; i < 16; i++) {
        all.add(stepCues([det('chair', cx: 0.5, depth: 0.7)]));
      }
      expect(all.any((c) => c.earcons.contains('arrived')), isTrue);
      expect(all.last.beacon, isNull);
    });

    test('no beacon without a goal', () {
      expect(stepCues([det('chair', cx: 0.5, depth: 3.0)]).beacon, isNull);
    });

    test('the beacon follows the planned path around an obstacle, not the straight line to the target', () {
      g.setGoal(SearchGoal(labels: ['chair'], phrase: 'a chair'));
      mapAt(1000, dets: [det('chair', cx: 0.5, depth: 3.0)]);
      mapAt(1300, dets: [det('chair', cx: 0.5, depth: 3.0)]); // found
      final c = mapAt(1600,
          dets: [det('chair', cx: 0.5, depth: 3.0)], p: plan(1600, hasGoal: true, steer: 20, gx: 0, gz: 3));
      expect(c.beacon, isNotNull);
      expect(c.beacon!.pan, greaterThan(0.5)); // 20 deg right of straight, though the chair is dead ahead
      expect(c.beacon!.locked, isFalse);
    });
  });

  group('directional hazard tones', () {
    test('an obstacle on the left is heard on the left, and only there', () {
      mapAt(1000, p: plan(1000, left: const SectorHit(1.0, -20)));
      final c = mapAt(1300, p: plan(1300, left: const SectorHit(1.0, -20)));
      expect(c.hazards[0], isNotNull);
      expect(c.hazards[0]!.pan, lessThan(-0.5));
      expect(c.hazards[1], isNull);
      expect(c.hazards[2], isNull);
    });

    test('an obstacle on the right is heard on the right', () {
      mapAt(1000, p: plan(1000, right: const SectorHit(1.2, 18)));
      final c = mapAt(1300, p: plan(1300, right: const SectorHit(1.2, 18)));
      expect(c.hazards[2], isNotNull);
      expect(c.hazards[2]!.pan, greaterThan(0.5));
      expect(c.hazards[0], isNull);
    });

    test('obstacles on both sides give two tones with opposite pans and different pitches', () {
      for (final at in [1000.0, 1300.0]) {
        mapAt(at, p: plan(at, left: const SectorHit(1.0, -20), right: const SectorHit(1.4, 15)));
      }
      final c = mapAt(1600, p: plan(1600, left: const SectorHit(1.0, -20), right: const SectorHit(1.4, 15)));
      expect(c.hazards[0]!.pan, lessThan(0));
      expect(c.hazards[2]!.pan, greaterThan(0));
      expect(c.hazards[0]!.hz, isNot(c.hazards[2]!.hz));
    });

    test('a single noisy plan does not sound; it must persist for two depth updates', () {
      final c = mapAt(1000, p: plan(1000, center: const SectorHit(1.0, 0)));
      expect(c.anyHazard, isFalse);
      expect(mapAt(1100, p: plan(1000, center: const SectorHit(1.0, 0))).anyHazard, isFalse); // same plan again
      expect(mapAt(1300, p: plan(1300, center: const SectorHit(1.0, 0))).anyHazard, isTrue);
    });

    test('rate and level rise with proximity; very close is a continuous tone', () {
      for (final at in [1000.0, 1300.0]) {
        mapAt(at, p: plan(at, center: const SectorHit(1.8, 0)));
      }
      final far = mapAt(1600, p: plan(1600, center: const SectorHit(1.8, 0))).hazards[1]!;
      final near = mapAt(1900, p: plan(1900, center: const SectorHit(0.8, 0))).hazards[1]!;
      final touching = mapAt(2200, p: plan(2200, center: const SectorHit(0.5, 0))).hazards[1]!;
      expect(near.ppm, greaterThan(far.ppm));
      expect(near.gain, greaterThan(far.gain));
      expect(touching.ppm, 0);
    });

    test('obstacles beyond the warning range are silent', () {
      for (final at in [1000.0, 1300.0, 1600.0]) {
        expect(mapAt(at, p: plan(at, center: const SectorHit(2.6, 0))).anyHazard, isFalse);
      }
    });

    test('the target itself is not a hazard, but something in front of it is', () {
      g.setGoal(SearchGoal(labels: ['chair'], phrase: 'a chair'));
      mapAt(500, dets: [det('chair', cx: 0.5, depth: 2.2)]);
      mapAt(800, dets: [det('chair', cx: 0.5, depth: 2.2)]); // found: guiding
      CueFrame c = const CueFrame();
      for (final at in [1100.0, 1400.0, 1700.0]) {
        c = mapAt(at,
            dets: [det('chair', cx: 0.5, depth: 2.2)],
            p: plan(at, center: const SectorHit(1.8, 0), hasGoal: true, steer: 0, gz: 2.2));
      }
      expect(c.hazards[1], isNull); // 1.8 m is the chair, which is where we are headed
      for (final at in [2000.0, 2300.0, 2600.0]) {
        c = mapAt(at,
            dets: [det('chair', cx: 0.5, depth: 2.2)],
            p: plan(at, center: const SectorHit(0.9, 0), hasGoal: true, steer: 0, gz: 2.2));
      }
      expect(c.hazards[1], isNotNull); // 0.9 m is well short of the chair: a real obstacle
    });

    test('the beacon is turned down while any hazard tone sounds', () {
      g.setGoal(SearchGoal(labels: ['chair'], phrase: 'a chair'));
      mapAt(500, dets: [det('chair', cx: 0.5, depth: 3.0)]);
      final free = mapAt(800, dets: [det('chair', cx: 0.5, depth: 3.0)]).beacon!;
      CueFrame c = const CueFrame();
      for (final at in [1100.0, 1400.0]) {
        c = mapAt(at,
            dets: [det('chair', cx: 0.5, depth: 3.0)], p: plan(at, left: const SectorHit(1.0, -20), gz: 3));
      }
      expect(c.anyHazard, isTrue);
      expect(c.beacon!.gain, lessThan(free.gain));
    });

    test('without a usable plan the coarse depth grid is the fallback, still directional', () {
      final grid = gridWith({(1, 0): 0.8}); // middle row, leftmost column
      mapAt(1000, grid: grid);
      final c = mapAt(1300, grid: grid);
      expect(c.hazards[0], isNotNull);
      expect(c.hazards[0]!.pan, lessThan(-0.5));
      expect(c.hazards[1], isNull);
      expect(c.hazards[2], isNull);
    });

    test('floor-level cells of the coarse grid are ignored in the fallback', () {
      final grid = gridWith({(2, 2): 0.5});
      mapAt(1000, grid: grid);
      expect(mapAt(1300, grid: grid).anyHazard, isFalse);
    });

    test('a stale plan is not used', () {
      mapAt(1000, p: plan(1000, left: const SectorHit(1.0, -20)));
      final c = mapAt(5000, p: plan(1000, left: const SectorHit(1.0, -20)));
      expect(c.anyHazard, isFalse);
    });
  });

  test('guided updates are not spoken by default (the beacon carries them)', () {
    g.setGoal(SearchGoal(labels: ['chair'], phrase: 'a chair'));
    final spoken = <Utterance>[];
    for (var i = 0; i < 30; i++) {
      spoken.addAll(step([det('chair', cx: 0.2 + 0.01 * i, depth: 4.0 - 0.05 * i)]));
    }
    // found is spoken once; no periodic "turn left / N meters" updates
    expect(spoken.where((u) => u.parts.first.startsWith('turn:') || u.parts.first == 'go:ahead'), isEmpty);
  });
}
