import 'dart:math' as math;

class CameraModel {
  const CameraModel({this.hfovDeg = 60});

  /// Horizontal field of view of the upright (portrait) frame, in degrees. 60 is a rough guess for a phone's
  /// main camera in portrait; read it from the camera characteristics or measure it before trusting bearings.
  final double hfovDeg;

  /// Degrees from the optical axis for a normalised x (0..1). Negative = left of centre.
  double bearingDeg(double xNorm) {
    final halfTan = math.tan(hfovDeg * math.pi / 360);
    return math.atan((xNorm - 0.5) * 2 * halfTan) * 180 / math.pi;
  }
}

enum Direction { farLeft, left, slightlyLeft, ahead, slightlyRight, right, farRight }

Direction directionFromBearing(double deg) {
  final a = deg.abs();
  final left = deg < 0;
  if (a < 7) return Direction.ahead;
  if (a < 15) return left ? Direction.slightlyLeft : Direction.slightlyRight;
  if (a < 25) return left ? Direction.left : Direction.right;
  return left ? Direction.farLeft : Direction.farRight;
}

String dirKey(Direction d) => switch (d) {
      Direction.ahead => 'dir:ahead',
      Direction.slightlyLeft => 'dir:slightly_left',
      Direction.left => 'dir:left',
      Direction.farLeft => 'dir:far_left',
      Direction.slightlyRight => 'dir:slightly_right',
      Direction.right => 'dir:right',
      Direction.farRight => 'dir:far_right',
    };

/// Spoken instruction for moving toward something in direction [d].
String turnKey(Direction d) => switch (d) {
      Direction.ahead => 'go:ahead',
      Direction.slightlyLeft => 'turn:slightly_left',
      Direction.left => 'turn:left',
      Direction.farLeft => 'turn:far_left',
      Direction.slightlyRight => 'turn:slightly_right',
      Direction.right => 'turn:right',
      Direction.farRight => 'turn:far_right',
    };

/// "I lost it" phrase naming the side it was last seen on.
String lostKey(Direction d) => switch (d) {
      Direction.ahead => 'lost:ahead',
      Direction.slightlyLeft || Direction.left || Direction.farLeft => 'lost:left',
      Direction.slightlyRight || Direction.right || Direction.farRight => 'lost:right',
    };

/// Plain words for scene summaries (text sent to Gemini, logs).
String directionWords(Direction d) => switch (d) {
      Direction.ahead => 'straight ahead',
      Direction.slightlyLeft => 'slightly left',
      Direction.left => 'left',
      Direction.farLeft => 'far left',
      Direction.slightlyRight => 'slightly right',
      Direction.right => 'right',
      Direction.farRight => 'far right',
    };

// Deliberately coarse: monocular depth is not trustworthy to better than about half a metre, so we never
// speak decimals. Buckets: <0.75 half, <1.25 one, <1.75 one and a half, <2.5 two, <3.5 three, <4.5 four,
// <5.5 five, <7 six, else far.
const _distEdges = <double>[0.75, 1.25, 1.75, 2.5, 3.5, 4.5, 5.5, 7.0];
const _distKeys = <String>[
  'dist:half', 'dist:1', 'dist:1_5', 'dist:2', 'dist:3', 'dist:4', 'dist:5', 'dist:6', 'dist:far',
];

int distanceBucket(double metres) {
  for (var i = 0; i < _distEdges.length; i++) {
    if (metres < _distEdges[i]) return i;
  }
  return _distEdges.length;
}

String distanceKey(double metres) => _distKeys[distanceBucket(metres)];
