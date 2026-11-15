import 'package:flutter/foundation.dart' show ValueNotifier;

import '../engine.dart';
import '../planning/planner.dart';
import 'cue_mapper.dart';
import 'escalation.dart';
import 'geometry.dart';
import 'guidance.dart';
import 'phrase_bank.dart';
import 'scene_tracker.dart';
import 'speech_arbiter.dart';

/// Wires the fused frame stream to tracking, path planning, guidance, escalation and speech. Pure logic: no plugins.
/// Gemini and audio hook in through [onEscalate], [setGoal] / [clearGoal] and [sceneSummary].
class DecisionEngine {
  DecisionEngine({
    required this.speech,
    GuidanceConfig? guidanceConfig,
    EscalationConfig? escalationConfig,
    this.onEscalate,
    this.canEscalate,
  })  : guidance = GuidanceEngine(guidanceConfig ?? const GuidanceConfig()),
        escalation = EscalationPolicy(escalationConfig ?? const EscalationConfig());

  final SceneTracker tracker = SceneTracker();
  final PathPlanner planner = PathPlanner();
  final GuidanceEngine guidance;
  final EscalationPolicy escalation;
  final SpeechArbiter speech;
  final CueMapper cueMapper = CueMapper();

  /// The latest path plan (updated whenever a frame carries a fresh depth grid). The path screen listens to it.
  final ValueNotifier<PlanResult?> plan = ValueNotifier<PlanResult?>(null);

  /// The scene is hard to understand locally: grab a snapshot and ask Gemini.
  void Function(EscalationReason reason, FusedFrame frame)? onEscalate;

  /// false while the user is already in a conversation with Gemini (then nothing escalates).
  bool Function()? canEscalate;

  /// Continuous sonification parameters, once per frame. Wire to CuePlayer.update.
  void Function(CueFrame cues)? onCues;

  FusedFrame? lastFrame;

  void onFrame(FusedFrame f) {
    lastFrame = f;
    final hf = f.hfovDeg;
    if (hf != null && (hf - guidance.camera.hfovDeg).abs() > 0.5) guidance.camera = CameraModel(hfovDeg: hf);

    tracker.update(f);
    final spoken = guidance.update(f, tracker);
    for (final u in spoken) {
      speech.submit(u);
    }
    _plan(f);
    onCues?.call(cueMapper.map(g: guidance, spoken: spoken, frame: f, plan: plan.value, nowMs: f.tMs));

    final reason = escalation.update(f: f, tracker: tracker, goal: guidance, allowed: canEscalate?.call() ?? true);
    if (reason != null) {
      speech.submit(const Utterance(['say:checking'],
          priority: Priority.status, key: 'checking', ttlMs: 2000));
      onEscalate?.call(reason, f);
    }
  }

  void _plan(FusedFrame f) {
    final grid = f.depthGrid;
    if (grid == null || grid.length != kGridW * kGridH) return;
    PlanGoal? goal;
    final t = guidance.activeTarget;
    final d = t?.depthMedian;
    if (t != null && guidance.phase == Phase.guiding && d != null && !d.isNaN) {
      goal = PlanGoal(t.cx, t.cy, d);
    }
    try {
      plan.value = planner.plan(grid, guidance.camera.hfovDeg, goal: goal, nowMs: f.tMs);
    } catch (_) {
      // A planner failure must never take the guidance loop down; the coarse depth-grid fallback takes over.
    }
  }

  /// [labels] are vocabulary words (the conversation layer maps the user's wording onto them).
  void setGoal(List<String> labels, String phrase) {
    if (labels.isEmpty) return;
    escalation.rearm();
    for (final u in guidance.setGoal(SearchGoal(labels: labels, phrase: phrase))) {
      speech.submit(u);
    }
  }

  void clearGoal() {
    for (final u in guidance.setGoal(null)) {
      speech.submit(u);
    }
  }

  /// Text description of what the local models currently see; Gemini can call this as a tool.
  String sceneSummary() {
    if (lastFrame == null) return 'No camera data yet.';
    final visible = tracker.confirmed.where(tracker.isVisible).toList()
      ..sort((a, b) => (a.depthMedian ?? double.infinity).compareTo(b.depthMedian ?? double.infinity));
    final p = plan.value;
    final ahead = (p != null && p.ok && p.obstacleAheadM != null)
        ? ' Something solid is about ${p.obstacleAheadM!.toStringAsFixed(1)} m straight ahead.'
        : '';
    if (visible.isEmpty) return 'No recognised objects in view.$ahead';
    final list = visible.map((t) {
      final dir = directionWords(directionFromBearing(guidance.camera.bearingDeg(t.cx)));
      final d = t.depthMedian;
      final dist = (d == null || d.isNaN) ? 'distance unknown' : 'about ${d.toStringAsFixed(1)} m';
      return '${t.label}: $dir, $dist';
    }).join('; ');
    return '$list.$ahead';
  }
}
