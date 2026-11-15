import 'dart:math' as math;

import '../engine.dart';
import 'guidance.dart';
import 'scene_tracker.dart';

enum EscalationReason { searchStalled, clutter, depthUnreliable }

class EscalationConfig {
  const EscalationConfig({
    this.enabled = true, // the user's consent to send camera snapshots off the device
    this.cooldownMs = 45000,
    this.maxPerSession = 3,
    this.stallMs = 20000,
    this.clutterTracks = 8,
    this.clutterHoldMs = 3000,
    this.farM = 9.0,
    this.farFraction = 0.6,
    this.depthHoldMs = 5000,
    this.rearmAfterClearMs = 5000,
  });

  final bool enabled;
  final double cooldownMs;
  final int maxPerSession;
  final double stallMs; // searching this long without a sighting
  final int clutterTracks; // this many confirmed objects at once...
  final double clutterHoldMs; // ...for this long
  final double farM; // a depth cell counts as "no useful depth" if invalid or beyond this
  final double farFraction; // ...and this share of the grid must be like that
  final double depthHoldMs;

  /// After a reason has fired it stays latched until its condition has been gone for this long. Without this a
  /// condition that persists (a cluttered room) would send a new snapshot every cooldown: Gemini would describe the
  /// same scene again and again.
  final double rearmAfterClearMs;
}

/// Decides when the local models are not enough and a snapshot should go to Gemini.
/// Only fires while a search goal is active: an empty or busy scene is only a problem when the user is looking
/// for something. Each reason fires once and then waits to be re-armed; everything is rate-limited.
class EscalationPolicy {
  EscalationPolicy([this.cfg = const EscalationConfig()]);

  final EscalationConfig cfg;
  int _count = 0;
  double _lastMs = -1e12;
  double? _clutterSince;
  double? _depthBadSince;
  final Set<EscalationReason> _latched = {};
  final Map<EscalationReason, double> _clearSince = {};

  int get count => _count;

  /// Full reset (a new session).
  void reset() {
    _count = 0;
    _lastMs = -1e12;
    _clutterSince = null;
    _depthBadSince = null;
    _latched.clear();
    _clearSince.clear();
  }

  /// A new search goal: every reason may fire again for it.
  void rearm() {
    _latched.clear();
    _clearSince.clear();
    _clutterSince = null;
    _depthBadSince = null;
  }

  /// [allowed] is false while the user is already talking to Gemini: nothing fires and nothing is consumed.
  EscalationReason? update({
    required FusedFrame f,
    required SceneTracker tracker,
    required GuidanceEngine goal,
    bool allowed = true,
  }) {
    if (!cfg.enabled || _count >= cfg.maxPerSession) return null;
    final now = f.tMs;

    // Timers keep running during the cooldown so a persistent problem fires as soon as it is allowed to.
    _clutterSince = tracker.confirmed.length >= cfg.clutterTracks ? (_clutterSince ?? now) : null;
    final grid = f.grid;
    if (grid != null && grid.isNotEmpty) {
      final bad = grid.where((v) => v.isNaN || v > cfg.farM).length / grid.length;
      _depthBadSince = bad >= cfg.farFraction ? (_depthBadSince ?? now) : null;
    }

    final searching = goal.goal != null;
    final since = math.max(goal.goal?.startedMs ?? now, goal.lastSightingMs ?? -1e12);
    final active = <EscalationReason, bool>{
      EscalationReason.searchStalled:
          searching && goal.phase == Phase.scanning && now - since >= cfg.stallMs,
      EscalationReason.clutter:
          searching && _clutterSince != null && now - _clutterSince! >= cfg.clutterHoldMs,
      EscalationReason.depthUnreliable:
          searching && _depthBadSince != null && now - _depthBadSince! >= cfg.depthHoldMs,
    };

    // Latched reasons re-arm once their condition has been gone for a while.
    for (final r in _latched.toList()) {
      if (active[r]!) {
        _clearSince.remove(r);
      } else {
        final since0 = _clearSince.putIfAbsent(r, () => now);
        if (now - since0 >= cfg.rearmAfterClearMs) {
          _latched.remove(r);
          _clearSince.remove(r);
        }
      }
    }

    if (!allowed || !searching || now - _lastMs < cfg.cooldownMs) return null;
    for (final r in EscalationReason.values) {
      if (active[r]! && !_latched.contains(r)) {
        _latched.add(r);
        _count++;
        _lastMs = now;
        return r;
      }
    }
    return null;
  }
}
