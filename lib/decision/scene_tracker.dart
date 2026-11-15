import '../engine.dart';

class Track {
  Track({
    required this.id,
    required this.label,
    required this.box,
    required this.score,
    required final tMs,
    this.depthMedian,
    this.depthNear,
  })  : firstSeenMs = tMs,
        lastSeenMs = tMs;

  final int id;
  final String label;
  List<double> box; // x1, y1, x2, y2, normalised, smoothed
  double score;
  double? depthMedian; // metres, smoothed
  double? depthNear; // nearer-biased, smoothed
  int hits = 1;
  double firstSeenMs;
  double lastSeenMs;

  double get cx => (box[0] + box[2]) / 2;
  double get cy => (box[1] + box[3]) / 2;
}

/// Turns per-frame detections into stable objects. Without this, guidance would flicker with every missed
/// or jittery detection at 2-7 fps.
class SceneTracker {
  SceneTracker({
    this.minHits = 2,
    this.maxAgeMs = 1500,
    this.iouMatch = 0.25,
    this.boxAlpha = 0.5,
    this.depthAlpha = 0.4,
  });

  final int minHits;
  final double maxAgeMs;
  final double iouMatch;
  final double boxAlpha;
  final double depthAlpha;

  final List<Track> tracks = [];
  double nowMs = 0;
  int _nextId = 1;

  /// Seen in the most recent frame (as opposed to coasting on its last known position).
  bool isVisible(Track t) => t.lastSeenMs >= nowMs;

  List<Track> get confirmed => tracks.where((t) => t.hits >= minHits).toList();

  void update(FusedFrame f) {
    nowMs = f.tMs;
    final dets = [...f.detections]..sort((a, b) => b.score.compareTo(a.score));
    final used = <int>{};
    for (final d in dets) {
      Track? best;
      var bestIou = iouMatch;
      for (final t in tracks) {
        if (used.contains(t.id) || t.label != d.label) continue;
        final v = _iou(t.box, d.box);
        if (v > bestIou) {
          bestIou = v;
          best = t;
        }
      }
      if (best == null) {
        final t = Track(
          id: _nextId++,
          label: d.label,
          box: List<double>.of(d.box),
          score: d.score,
          tMs: f.tMs,
          depthMedian: _valid(d.depthMedian),
          depthNear: _valid(d.depthP10),
        );
        tracks.add(t);
        used.add(t.id);
      } else {
        for (var i = 0; i < 4; i++) {
          best.box[i] += boxAlpha * (d.box[i] - best.box[i]);
        }
        best.score += boxAlpha * (d.score - best.score);
        best.depthMedian = _blend(best.depthMedian, _valid(d.depthMedian));
        best.depthNear = _blend(best.depthNear, _valid(d.depthP10));
        best.hits++;
        best.lastSeenMs = f.tMs;
        used.add(best.id);
      }
    }
    tracks.removeWhere((t) => f.tMs - t.lastSeenMs > maxAgeMs);
  }

  double? _valid(double? v) => (v == null || v.isNaN) ? null : v;

  double? _blend(double? old, double? v) {
    if (v == null) return old;
    if (old == null) return v;
    return old + depthAlpha * (v - old);
  }

  static double _iou(List<double> a, List<double> b) {
    final iw = (a[2] < b[2] ? a[2] : b[2]) - (a[0] > b[0] ? a[0] : b[0]);
    final ih = (a[3] < b[3] ? a[3] : b[3]) - (a[1] > b[1] ? a[1] : b[1]);
    if (iw <= 0 || ih <= 0) return 0;
    final inter = iw * ih;
    final union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return union > 0 ? inter / union : 0;
  }
}
