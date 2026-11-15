import 'dart:convert';

import 'package:flutter/services.dart' show rootBundle;

/// Higher index = more urgent. Hazards preempt everything, including Gemini's voice.
enum Priority { status, guidance, hazard }

/// One spoken message, built from phrase-bank keys so it can be played from pre-rendered Kokoro clips
/// (no on-device synthesis latency).
class Utterance {
  const Utterance(this.parts, {required this.priority, this.key, this.ttlMs = 4000, this.urgent = false});

  final List<String> parts;
  final Priority priority;

  /// A newer utterance with the same key replaces an older one still waiting in the queue.
  final String? key;

  /// Dropped if it has not started within this time: stale guidance is worse than silence.
  final int ttlMs;

  /// Only urgent utterances may cut across Gemini's voice. Everything else waits for it to finish (hazards are
  /// also carried by the directional tones, which are never held).
  final bool urgent;

  @override
  String toString() => 'Utterance(${priority.name}, $parts)';
}

class PhraseBank {
  PhraseBank(this._fixed);

  final Map<String, String> _fixed;

  static Future<PhraseBank> load() async {
    final s = await rootBundle.loadString('assets/phrase_bank/phrases.json');
    return PhraseBank(Map<String, String>.from(jsonDecode(s) as Map));
  }

  static String slug(String label) => label
      .toLowerCase()
      .replaceAll(RegExp(r'[^a-z0-9]+'), '_')
      .replaceAll(RegExp(r'^_+|_+$'), '');

  static String labelKey(String label) => 'label:${slug(label)}';

  bool has(String key) => key.startsWith('label:') || _fixed.containsKey(key);

  String text(String key) =>
      key.startsWith('label:') ? key.substring(6).replaceAll('_', ' ') : (_fixed[key] ?? key);

  String render(List<String> parts) => parts.map(text).join(', ');
}
