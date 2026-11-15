import 'dart:math' as math;

import '../engine.dart';
import 'geometry.dart';
import 'phrase_bank.dart';
import 'scene_tracker.dart';

enum Phase { idle, scanning, guiding, arrived }

class SearchGoal {
  SearchGoal({required this.labels, required this.phrase});

  /// Vocabulary labels that satisfy the request (the first one is spoken).
  final List<String> labels;

  /// What the user actually said, for logs.
  final String phrase;

  /// Set from the first frame after the goal is created (frame time, not wall-clock).
  double? startedMs;
}

class GuidanceConfig {
  const GuidanceConfig({
    this.camera = const CameraModel(),
    this.minGapMs = 2000,
    this.heartbeatMs = 6000,
    this.scanPromptMs = 8000,
    this.lostAfterMs = 2500,
    this.arrivalM = 0.9,
    this.arrivalBearingDeg = 20,
    this.hazardLabels = const {'stairs', 'car', 'bus', 'truck', 'bicycle', 'motorcycle'},
    this.hazardRangeM = 3.5,
    this.hazardCooldownMs = 8000,
    this.corridorEnabled = true,
    this.corridorWarnM = 1.2,
    this.corridorStopM = 0.7,
    this.corridorCooldownMs = 3000,
    this.spokenUpdates = false,
    this.corridorSpeech = false,
    this.corridorStopSpeech = true,
  });

  final CameraModel camera;
  final double minGapMs; // minimum time between guidance updates when something changed
  final double heartbeatMs; // restate guidance at least this often while guiding
  final double scanPromptMs; // how often to remind the user to keep turning while searching
  final double lostAfterMs;
  final double arrivalM;
  final double arrivalBearingDeg;
  final Set<String> hazardLabels; // must be words in the model vocabulary to ever fire
  final double hazardRangeM;
  final double hazardCooldownMs;

  /// Experimental. Looks at the middle row of the depth grid. It cannot see low obstacles, steps or drop-offs,
  /// and it assumes the phone is held roughly upright (tilted down, the floor fills the middle row).
  final bool corridorEnabled;
  final double corridorWarnM;
  final double corridorStopM;
  final double corridorCooldownMs;

  /// Speak 'turn left, two meters' style updates while guiding. Off by default: the sonified beacon carries
  /// direction and distance, and speech is kept for discrete events (found, lost, arrived, hazard class).
  final bool spokenUpdates;

  /// Speak 'something close ahead' (the hazard tone already says so) / a one-shot 'Stop' when very close.
  final bool corridorSpeech;
  final bool corridorStopSpeech;
}

/// The search state machine: idle -> scanning -> guiding -> arrived, plus always-on hazard warnings.
/// Produces utterances; it never speaks by itself.
class GuidanceEngine {
  GuidanceEngine([this.cfg = const GuidanceConfig()]) : camera = cfg.camera;

  final GuidanceConfig cfg;

  /// Camera geometry used for bearings. Starts from the config; the decision engine replaces it with the
  /// field of view the engine read from the camera.
  CameraModel camera;
  Phase phase = Phase.idle;
  SearchGoal? goal;
  double? lastSightingMs;

  // Read by CueMapper after each update().
  Track? activeTarget; // the target while guiding and visible, else null
  double? corridorNearest; // nearest valid depth (m) in the centre of the middle depth-grid row
  double corridorPan = 0; // -0.6 / 0 / +0.6: which of the three centre columns is nearest
  bool corridorActive = false; // below the warning distance for two consecutive frames

  int? _targetId;
  Direction? _lastDir;
  int? _lastBucket;
  double _lastGuideMs = -1e12;
  double _lastScanMs = -1e12;
  final Map<String, double> _hazardAt = {};
  double _corridorAt = -1e12;
  int _corridorFrames = 0;

  List<Utterance> setGoal(SearchGoal? g) {
    goal = g;
    _targetId = null;
    lastSightingMs = null;
    _lastDir = null;
    _lastBucket = null;
    _lastGuideMs = -1e12;
    _lastScanMs = -1e12;
    if (g == null) {
      phase = Phase.idle;
      return const [Utterance(['say:search_cancelled'], priority: Priority.status, key: 'goal')];
    }
    phase = Phase.scanning;
    return [
      Utterance(['say:looking_for', PhraseBank.labelKey(g.labels.first)],
          priority: Priority.guidance, key: 'goal'),
    ];
  }

  List<Utterance> update(FusedFrame f, SceneTracker tr) {
    final now = f.tMs;
    activeTarget = null;
    final out = <Utterance>[..._hazards(f, tr, now)];
    final g = goal;
    if (g == null || phase == Phase.idle) return out;
    g.startedMs ??= now;

    final target = _pickTarget(tr, g);
    switch (phase) {
      case Phase.scanning:
        if (target != null && tr.isVisible(target)) {
          phase = Phase.guiding;
          _targetId = target.id;
          activeTarget = target;
          lastSightingMs = now;
          final dir = _dir(target);
          _lastDir = dir;
          _lastBucket = _bucket(target);
          _lastGuideMs = now;
          out.add(Utterance(
            ['say:found', PhraseBank.labelKey(target.label), dirKey(dir), ..._distParts(target)],
            priority: Priority.guidance,
            key: 'target',
            ttlMs: 3000,
          ));
        } else if (now - math.max(_lastScanMs, g.startedMs!) >= cfg.scanPromptMs) {
          _lastScanMs = now;
          out.add(const Utterance(['say:keep_scanning'],
              priority: Priority.status, key: 'scan', ttlMs: 2000));
        }
      case Phase.guiding:
        if (target != null && tr.isVisible(target)) {
          lastSightingMs = now;
          _targetId = target.id;
          activeTarget = target;
          final dir = _dir(target);
          final near = _num(target.depthNear) ?? _num(target.depthMedian);
          final bearing = camera.bearingDeg(target.cx).abs();
          if (near != null && near <= cfg.arrivalM && bearing < cfg.arrivalBearingDeg) {
            phase = Phase.arrived;
            out.add(const Utterance(['say:arrived'],
                priority: Priority.guidance, key: 'target', ttlMs: 3000));
            break;
          }
          final bucket = _bucket(target);
          final changed = dir != _lastDir || bucket != _lastBucket;
          final since = now - _lastGuideMs;
          if ((changed && since >= cfg.minGapMs) || since >= cfg.heartbeatMs) {
            _lastDir = dir;
            _lastBucket = bucket;
            _lastGuideMs = now;
            if (cfg.spokenUpdates) {
              out.add(Utterance([turnKey(dir), ..._distParts(target)],
                  priority: Priority.guidance, key: 'target', ttlMs: 2500));
            }
          }
        } else if (lastSightingMs != null && now - lastSightingMs! > cfg.lostAfterMs) {
          phase = Phase.scanning;
          _targetId = null;
          _lastScanMs = now;
          out.add(Utterance([lostKey(_lastDir ?? Direction.ahead)],
              priority: Priority.guidance, key: 'target', ttlMs: 3000));
        }
      case Phase.arrived:
        // Silent. If the target is clearly far again (it moved, or we walked away), resume guiding.
        if (target != null && tr.isVisible(target)) {
          lastSightingMs = now;
          final near = _num(target.depthNear) ?? _num(target.depthMedian);
          if (near != null && near > cfg.arrivalM + 0.8) {
            phase = Phase.guiding;
            _lastGuideMs = -1e12;
            _lastDir = null;
            _lastBucket = null;
          }
        }
      case Phase.idle:
        break;
    }
    return out;
  }

  List<Utterance> _hazards(FusedFrame f, SceneTracker tr, double now) {
    final out = <Utterance>[];
    corridorNearest = null;
    corridorActive = false;

    for (final t in tr.confirmed) {
      if (!cfg.hazardLabels.contains(t.label) || !tr.isVisible(t)) continue;
      if (goal != null && goal!.labels.contains(t.label)) continue; // that is the target, guidance covers it
      final near = _num(t.depthNear) ?? _num(t.depthMedian);
      if (near == null || near > cfg.hazardRangeM) continue;
      final bearing = camera.bearingDeg(t.cx);
      if (bearing.abs() > 25) continue;
      if (now - (_hazardAt[t.label] ?? -1e12) < cfg.hazardCooldownMs) continue;
      _hazardAt[t.label] = now;
      out.add(Utterance(
        ['say:careful', PhraseBank.labelKey(t.label), dirKey(directionFromBearing(bearing)), distanceKey(near)],
        priority: Priority.hazard,
        key: 'hazard:${t.label}',
        ttlMs: 1500,
      ));
    }

    final grid = f.grid;
    if (cfg.corridorEnabled && grid != null && f.gridRows >= 2 && f.gridCols >= 3) {
      final row = f.gridRows ~/ 2;
      final c0 = (f.gridCols - 3) ~/ 2;
      double? nearest;
      var nearestCol = c0 + 1;
      for (var c = c0; c < c0 + 3; c++) {
        final v = grid[row * f.gridCols + c];
        if (v.isNaN) continue;
        if (nearest == null || v < nearest) {
          nearest = v;
          nearestCol = c;
        }
      }
      corridorNearest = nearest;
      corridorPan = (nearestCol - (c0 + 1)) * 0.6;
      if (nearest != null && nearest < cfg.corridorWarnM) {
        _corridorFrames++; // require two consecutive frames: one noisy depth map must not trigger a warning
        corridorActive = _corridorFrames >= 2;
        if (corridorActive && now - _corridorAt >= cfg.corridorCooldownMs) {
          _corridorAt = now;
          if (nearest < cfg.corridorStopM) {
            if (cfg.corridorStopSpeech) {
              out.add(const Utterance(['say:stop'],
                  priority: Priority.hazard, key: 'corridor', ttlMs: 1000, urgent: true));
            }
          } else if (cfg.corridorSpeech) {
            out.add(Utterance(['say:something_close', distanceKey(nearest)],
                priority: Priority.hazard, key: 'corridor', ttlMs: 1500));
          }
        }
      } else {
        _corridorFrames = 0;
      }
    }
    return out;
  }

  /// Sticky: keep following the same track while it exists; otherwise the nearest confirmed candidate.
  Track? _pickTarget(SceneTracker tr, SearchGoal g) {
    final cands = tr.confirmed.where((t) => g.labels.contains(t.label)).toList();
    if (cands.isEmpty) return null;
    for (final t in cands) {
      if (t.id == _targetId) return t;
    }
    cands.sort((a, b) => (_num(a.depthMedian) ?? double.infinity)
        .compareTo(_num(b.depthMedian) ?? double.infinity));
    return cands.first;
  }

  Direction _dir(Track t) => directionFromBearing(camera.bearingDeg(t.cx));
  double? _num(double? v) => (v == null || v.isNaN) ? null : v;
  int _bucket(Track t) {
    final d = _num(t.depthMedian);
    return d == null ? -1 : distanceBucket(d);
  }

  List<String> _distParts(Track t) {
    final d = _num(t.depthMedian);
    return d == null ? const [] : [distanceKey(d)];
  }
}
