import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:spatial_nav/decision/escalation.dart';
import 'package:spatial_nav/decision/geometry.dart';
import 'package:spatial_nav/decision/guidance.dart';
import 'package:spatial_nav/decision/phrase_bank.dart';
import 'package:spatial_nav/decision/scene_tracker.dart';
import 'package:spatial_nav/decision/speech_arbiter.dart';

import 'helpers.dart';

class FakeOutput implements SpeechOutput {
  final spoken = <List<String>>[];
  var stops = 0;

  @override
  Future<void> speak(List<String> parts) async => spoken.add(parts);

  @override
  Future<void> stop() async => stops++;
}

class SlowOutput implements SpeechOutput {
  final spoken = <List<String>>[];
  var stops = 0;
  Completer? _c;

  @override
  Future<void> speak(List<String> parts) {
    spoken.add(parts);
    _c = Completer<void>();
    return _c!.future;
  }

  @override
  Future<void> stop() async {
    stops++;
    _c?.complete();
  }

  void finish() => _c?.complete();
}

Future<void> settle() async {
  for (var i = 0; i < 10; i++) {
    await Future<void>.delayed(Duration.zero);
  }
}

void main() {
  group('geometry', () {
    test('bearing', () {
      const cam = CameraModel(hfovDeg: 60);
      expect(cam.bearingDeg(0.5), closeTo(0, 1e-9));
      expect(cam.bearingDeg(0), closeTo(-30, 1e-6));
      expect(cam.bearingDeg(1), closeTo(30, 1e-6));
    });

    test('directions', () {
      expect(directionFromBearing(0), Direction.ahead);
      expect(directionFromBearing(-10), Direction.slightlyLeft);
      expect(directionFromBearing(20), Direction.right);
      expect(directionFromBearing(-30), Direction.farLeft);
    });

    test('distance buckets are coarse', () {
      expect(distanceKey(0.5), 'dist:half');
      expect(distanceKey(2.0), 'dist:2');
      expect(distanceKey(6.0), 'dist:6');
      expect(distanceKey(8.0), 'dist:far');
    });
  });

  group('tracker', () {
    test('confirms after two hits and expires by time', () {
      final tr = SceneTracker();
      tr.update(frame(1000, dets: [det('chair', depth: 2)]));
      expect(tr.confirmed, isEmpty);
      tr.update(frame(1300, dets: [det('chair', depth: 2)]));
      expect(tr.confirmed.length, 1);
      tr.update(frame(2000)); // coasting
      expect(tr.confirmed.length, 1);
      expect(tr.isVisible(tr.confirmed.first), isFalse);
      tr.update(frame(3000));
      expect(tr.tracks, isEmpty);
    });
  });

  group('guidance', () {
    late SceneTracker tr;
    late GuidanceEngine g;
    var t = 1000.0;
    List<Utterance> step(List<DetectionObj> dets, {List<double>? grid}) {
      final f = frame(t, dets: dets, grid: grid);
      t += 300;
      tr.update(f);
      return g.update(f, tr);
    }

    setUp(() {
      tr = SceneTracker();
      g = GuidanceEngine();
      t = 1000.0;
    });

    test('scan, found, arrive', () {
      g.setGoal(SearchGoal(labels: ['chair'], phrase: 'a chair'));
      expect(step([det('chair', cx: 0.2, depth: 3.0)]), isEmpty); // not confirmed yet
      final found = step([det('chair', cx: 0.2, depth: 3.0)]);
      expect(found.single.parts.first, 'say:found');
      expect(found.single.parts, containsAll(['label:chair', 'dir:left']));
      final all = <Utterance>[];
      for (var i = 0; i < 12; i++) {
        all.addAll(step([det('chair', cx: 0.5, depth: 0.7)]));
      }
      expect(all.any((u) => u.parts.contains('say:arrived')), isTrue);
      expect(g.phase, Phase.arrived);
    });

    test('lost target returns to scanning with a side hint', () {
      g.setGoal(SearchGoal(labels: ['chair'], phrase: 'a chair'));
      step([det('chair', cx: 0.2, depth: 3.0)]);
      step([det('chair', cx: 0.2, depth: 3.0)]);
      final all = <Utterance>[];
      for (var i = 0; i < 14; i++) {
        all.addAll(step(const []));
      }
      expect(all.any((u) => u.parts.first == 'lost:left'), isTrue);
      expect(g.phase, Phase.scanning);
    });

    test('hazard label speaks once within its cooldown', () {
      final all = <Utterance>[];
      for (var i = 0; i < 8; i++) {
        all.addAll(step([det('stairs', depth: 2.0)]));
      }
      final hazards = all.where((u) => u.priority == Priority.hazard).toList();
      expect(hazards.length, 1);
      expect(hazards.single.parts, containsAll(['say:careful', 'label:stairs']));
    });

    test('corridor warning needs two consecutive frames (speech enabled)', () {
      g = GuidanceEngine(const GuidanceConfig(corridorSpeech: true));
      final grid = gridWith({(1, 2): 0.9});
      expect(step(const [], grid: grid), isEmpty);
      final second = step(const [], grid: grid);
      expect(second.single.parts, ['say:something_close', 'dist:1']);
    });

    test('floor-level cells do not trigger the corridor warning', () {
      final grid = gridWith({(2, 2): 0.6});
      expect(step(const [], grid: grid), isEmpty);
      expect(step(const [], grid: grid), isEmpty);
    });
  });

  group('escalation', () {
    late SceneTracker tr;
    late GuidanceEngine g;

    EscalationReason? tick(EscalationPolicy esc, double at, {bool allowed = true}) {
      final f = frame(at);
      tr.update(f);
      return esc.update(f: f, tracker: tr, goal: g, allowed: allowed);
    }

    setUp(() {
      tr = SceneTracker();
      g = GuidanceEngine();
      g.setGoal(SearchGoal(labels: ['keys'], phrase: 'keys'));
      final f = frame(1000);
      tr.update(f);
      g.update(f, tr); // sets startedMs
    });

    test('fires after a stalled search, once, even though the condition persists', () {
      final esc = EscalationPolicy();
      expect(tick(esc, 1000), isNull);
      expect(tick(esc, 1000 + 21000), EscalationReason.searchStalled);
      expect(tick(esc, 1000 + 30000), isNull); // cooldown
      expect(tick(esc, 1000 + 90000), isNull); // cooldown is over but the reason is latched: no repeat
    });

    test('a new search goal re-arms it', () {
      final esc = EscalationPolicy();
      tick(esc, 1000);
      expect(tick(esc, 22000), EscalationReason.searchStalled);
      esc.rearm();
      expect(tick(esc, 80000), EscalationReason.searchStalled);
    });

    test('session cap', () {
      final esc = EscalationPolicy(const EscalationConfig(maxPerSession: 1, cooldownMs: 1000));
      tick(esc, 1000);
      expect(tick(esc, 22000), EscalationReason.searchStalled);
      esc.rearm();
      expect(tick(esc, 60000), isNull);
    });

    test('while the user is already talking to Gemini nothing fires and nothing is used up', () {
      final esc = EscalationPolicy();
      tick(esc, 1000);
      expect(tick(esc, 22000, allowed: false), isNull);
      expect(esc.count, 0);
      expect(tick(esc, 23000), EscalationReason.searchStalled);
    });

    test('never fires without a goal', () {
      final tr2 = SceneTracker();
      final g2 = GuidanceEngine();
      final esc = EscalationPolicy();
      final f = frame(100000, grid: List<double>.filled(15, double.nan));
      expect(esc.update(f: f, tracker: tr2, goal: g2), isNull);
    });
  });

  group('speech arbiter', () {
    test('priority order and replace-by-key', () async {
      final out = FakeOutput();
      final a = SpeechArbiter(out);
      a.hold();
      a.submit(const Utterance(['status'], priority: Priority.status));
      a.submit(const Utterance(['g1'], priority: Priority.guidance, key: 'target'));
      a.submit(const Utterance(['g2'], priority: Priority.guidance, key: 'target')); // replaces g1
      a.release();
      await settle();
      expect(out.spoken, [
        ['g2'],
        ['status'],
      ]);
    });

    test('stale utterances are dropped', () async {
      var now = 0.0;
      final out = FakeOutput();
      final a = SpeechArbiter(out, nowMs: () => now);
      a.hold();
      a.submit(const Utterance(['old'], priority: Priority.guidance, ttlMs: 1000));
      now = 5000;
      a.release();
      await settle();
      expect(out.spoken, isEmpty);
    });

    test('a hazard preempts what is playing and goes first', () async {
      final out = SlowOutput();
      final a = SpeechArbiter(out);
      a.submit(const Utterance(['guide'], priority: Priority.guidance));
      await settle();
      expect(out.spoken, [
        ['guide']
      ]);
      a.submit(const Utterance(['stop'], priority: Priority.hazard));
      await settle();
      expect(out.stops, 1);
      expect(out.spoken.last, ['stop']);
      out.finish();
    });

    test('hold blocks everything except urgent utterances', () async {
      var preempted = 0;
      final out = FakeOutput();
      final a = SpeechArbiter(out, onPreemptWhileHeld: () => preempted++);
      a.hold();
      a.submit(const Utterance(['guide'], priority: Priority.guidance));
      a.submit(const Utterance(['careful'], priority: Priority.hazard)); // a hazard, but not urgent: it waits
      a.submit(const Utterance(['stop'], priority: Priority.hazard, urgent: true));
      await settle();
      expect(out.spoken, [
        ['stop']
      ]);
      expect(preempted, 1);
      a.release();
      await settle();
      expect(out.spoken.map((p) => p.first), ['stop', 'careful', 'guide']);
    });

    test('urgent utterances cannot keep cutting Gemini off', () async {
      var now = 0.0;
      var preempted = 0;
      final out = FakeOutput();
      final a = SpeechArbiter(out, nowMs: () => now, onPreemptWhileHeld: () => preempted++, preemptCooldownMs: 8000);
      a.hold();
      a.submit(const Utterance(['stop1'], priority: Priority.hazard, urgent: true, key: 'a'));
      await settle();
      now = 2000;
      a.submit(const Utterance(['stop2'], priority: Priority.hazard, urgent: true, key: 'b')); // within the cool-down
      await settle();
      expect(preempted, 1);
      expect(out.spoken.length, 1);
      now = 9000;
      a.submit(const Utterance(['stop3'], priority: Priority.hazard, urgent: true, key: 'c'));
      await settle();
      expect(preempted, 2);
      expect(out.spoken.length, 2);
    });
  });

  group('phrase bank', () {
    test('slug and label keys', () {
      expect(PhraseBank.labelKey('Traffic Light'), 'label:traffic_light');
      expect(PhraseBank(const {}).text('label:traffic_light'), 'traffic light');
    });
  });
}
