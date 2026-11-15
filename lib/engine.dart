import 'dart:typed_data';

import 'package:flutter/services.dart';

/// Model/cadence settings sent to the native engine. Presets let you compare variants on-device without rebuilding.
class EngineConfig {
  const EngineConfig({
    required this.name,
    required this.yoloeAsset,
    required this.depthAsset,
    required this.depthSize,
    this.yoloeMaskCoefs = 0,
    this.depthEvery = 1,
    this.confThr = 0.25,
    this.iouThr = 0.7,
    this.overlap = true,
  });

  final String name;
  final String yoloeAsset;
  final int yoloeMaskCoefs; // 0 for the detect-only exports
  final String depthAsset;
  final int depthSize;
  final int depthEvery; // depth runs on every Nth frame
  final double confThr;
  final double iouThr;
  final bool overlap;

  Map<String, dynamic> toMap() => {
        'yoloeAsset': yoloeAsset,
        'yoloeMaskCoefs': yoloeMaskCoefs,
        'depthAsset': depthAsset,
        'depthSize': depthSize,
        'depthEvery': depthEvery,
        'confThr': confThr,
        'iouThr': iouThr,
        'overlap': overlap,
      };

  static const presets = <EngineConfig>[
    EngineConfig(
      name: 'quality: 26s-det + depth s-768',
      yoloeAsset: 'yoloe-26s-det.tflite',
      depthAsset: 'yolo26s-depth-768.tflite',
      depthSize: 768,
    ),
    EngineConfig(
      name: 'balanced: 26s-det + depth s-640',
      yoloeAsset: 'yoloe-26s-det.tflite',
      depthAsset: 'yolo26s-depth-640.tflite',
      depthSize: 640,
    ),
    EngineConfig(
      name: 'fast: 26n-det + depth s-640',
      yoloeAsset: 'yoloe-26n-det.tflite',
      depthAsset: 'yolo26s-depth-640.tflite',
      depthSize: 640,
    ),
    EngineConfig(
      name: 'quality, no overlap (A/B test)',
      yoloeAsset: 'yoloe-26s-det.tflite',
      depthAsset: 'yolo26s-depth-768.tflite',
      depthSize: 768,
      overlap: false,
    ),
    EngineConfig(
      name: 'quality, depth every 3rd frame',
      yoloeAsset: 'yoloe-26s-det.tflite',
      depthAsset: 'yolo26s-depth-768.tflite',
      depthSize: 768,
      depthEvery: 3,
    ),
  ];
}

class DetectionObj {
  DetectionObj({
    required this.label,
    required this.score,
    required this.box,
    this.depthMedian,
    this.depthP10,
  });

  final String label;
  final double score;

  /// x1, y1, x2, y2 normalised 0..1 in the upright camera frame.
  final List<double> box;

  /// Metres, from the latest depth map (see FusedFrame.depthAgeMs). NaN if no valid samples.
  final double? depthMedian;
  final double? depthP10; // nearer-biased

  factory DetectionObj.fromMap(Map m) => DetectionObj(
        label: m['label'] as String,
        score: (m['score'] as num).toDouble(),
        box: (m['box'] as List).map((e) => (e as num).toDouble()).toList(),
        depthMedian: (m['depthMedian'] as num?)?.toDouble(),
        depthP10: (m['depthP10'] as num?)?.toDouble(),
      );
}

/// The combined output of both models for one camera frame: the input to the decision engine.
class FusedFrame {
  FusedFrame({
    required this.frameId,
    required this.tMs,
    required this.w,
    required this.h,
    required this.detections,
    required this.depthFresh,
    required this.depthAgeMs,
    required this.grid,
    required this.gridRows,
    required this.gridCols,
    required this.timing,
    this.depthGrid,
    this.hfovDeg,
    this.previewJpeg,
  });

  final int frameId;
  final double tMs; // camera sensor timestamp
  final int w, h; // upright frame size in pixels
  final List<DetectionObj> detections;
  final bool depthFresh; // depth was computed from this very frame
  final double depthAgeMs; // -1 if no depth yet
  final List<double>? grid; // row-major gridRows x gridCols, 10th-percentile depth per cell (m)
  final int gridRows, gridCols;
  final Map<String, double> timing;

  /// kGridW x kGridH metres (row-major, row 0 = top of the upright frame, NaN = invalid). Only on frames that carry
  /// a fresh depth map; this is what the path planner consumes.
  final Float32List? depthGrid;

  /// Horizontal field of view of the upright frame, read from the camera (null = unknown).
  final double? hfovDeg;

  /// Small JPEG of the upright frame; only present while a visualisation screen has asked for it.
  final Uint8List? previewJpeg;

  factory FusedFrame.fromMap(Map m) => FusedFrame(
        frameId: (m['frameId'] as num).toInt(),
        tMs: (m['tMs'] as num).toDouble(),
        w: (m['w'] as num).toInt(),
        h: (m['h'] as num).toInt(),
        detections: (m['detections'] as List).map((e) => DetectionObj.fromMap(e as Map)).toList(),
        depthFresh: m['depthFresh'] as bool,
        depthAgeMs: (m['depthAgeMs'] as num).toDouble(),
        grid: (m['grid'] as List?)?.map((e) => (e as num).toDouble()).toList(),
        gridRows: (m['gridRows'] as num?)?.toInt() ?? 0,
        gridCols: (m['gridCols'] as num?)?.toInt() ?? 0,
        timing: (m['timing'] as Map).map((k, v) => MapEntry(k as String, (v as num).toDouble())),
        depthGrid: _floats(m['depthGrid']),
        hfovDeg: (m['hfovDeg'] as num?)?.toDouble(),
        previewJpeg: m['previewJpeg'] as Uint8List?,
      );
}

Float32List? _floats(Object? v) {
  if (v == null) return null;
  if (v is Float32List) return v;
  if (v is List) return Float32List.fromList([for (final x in v) (x as num).toDouble()]);
  return null;
}

class SceneEngine {
  static const _method = MethodChannel('spatial_nav/engine');
  static const _events = EventChannel('spatial_nav/engine_events');
  static const _keys = EventChannel('spatial_nav/keys');

  Stream<FusedFrame> frames() =>
      _events.receiveBroadcastStream().map((e) => FusedFrame.fromMap(e as Map));

  /// Returns once the camera permission is settled; model loading continues natively and the first frame
  /// arrives when the models are ready.
  Future<void> start(EngineConfig c) => _method.invokeMethod('start', c.toMap());

  Future<void> stop() => _method.invokeMethod('stop');

  /// Visualisation screens turn this on while visible: every frame then carries a JPEG preview.
  Future<void> setViz(bool on) => _method.invokeMethod<void>('viz', on);

  /// While true the hardware volume keys are delivered on [volumeKeys] ('up' / 'down') and do not change the volume.
  Future<void> setCaptureVolumeKeys(bool on) => _method.invokeMethod<void>('captureVolumeKeys', on);

  Stream<String> get volumeKeys => _keys.receiveBroadcastStream().map((e) => e as String);

  /// JPEG of the latest upright camera frame (null if the engine is not running). This is what goes to Gemini.
  Future<Uint8List?> snapshot() => _method.invokeMethod<Uint8List>('snapshot');
}
