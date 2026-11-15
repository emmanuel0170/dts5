import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;

import '../decision/speech_arbiter.dart';
import 'voice_client.dart';

const kClipSampleRate = 24000;

/// Extracts the PCM from a 16-bit mono 24 kHz WAV (what tools/render_phrase_bank.py writes).
/// Walks the RIFF chunks rather than assuming a 44-byte header. Throws FormatException on anything else,
/// so a wrongly rendered clip can never be played as noise.
Uint8List wavPcm(Uint8List wav) {
  final bd = ByteData.sublistView(wav);
  String tag(int o) => String.fromCharCodes(wav.sublist(o, o + 4));
  if (wav.length < 12 || tag(0) != 'RIFF' || tag(8) != 'WAVE') throw const FormatException('not a WAV file');
  var o = 12;
  var fmtOk = false;
  while (o + 8 <= wav.length) {
    final id = tag(o);
    final size = bd.getUint32(o + 4, Endian.little);
    final body = o + 8;
    if (id == 'fmt ') {
      final format = bd.getUint16(body, Endian.little);
      final channels = bd.getUint16(body + 2, Endian.little);
      final rate = bd.getUint32(body + 4, Endian.little);
      final bits = bd.getUint16(body + 14, Endian.little);
      if (format != 1 || channels != 1 || rate != kClipSampleRate || bits != 16) {
        throw FormatException('expected 16-bit mono PCM at $kClipSampleRate Hz, got format=$format '
            'channels=$channels rate=$rate bits=$bits');
      }
      fmtOk = true;
    } else if (id == 'data') {
      if (!fmtOk) throw const FormatException('data chunk before fmt chunk');
      final end = (body + size) > wav.length ? wav.length : body + size;
      return Uint8List.sublistView(wav, body, end);
    }
    o = body + size + (size.isOdd ? 1 : 0);
  }
  throw const FormatException('no data chunk');
}

/// The pre-rendered Kokoro phrase bank, decoded into memory (about 10 MB). Built from phrase-bank keys, so a
/// spoken message is a concatenation of clips with short gaps: zero synthesis latency, no CPU cost.
class PhraseClips {
  PhraseClips(this._pcm);

  final Map<String, Uint8List> _pcm;

  int get count => _pcm.length;

  /// Missing manifest (the phrase bank has not been rendered yet) gives an empty bank: guidance stays silent
  /// but the sonification cues and Gemini still work.
  static Future<PhraseClips> load() async {
    try {
      final manifest =
          jsonDecode(await rootBundle.loadString('assets/phrase_bank/manifest.json')) as Map<String, dynamic>;
      final out = <String, Uint8List>{};
      for (final e in manifest.entries) {
        final file = (e.value as Map)['file'] as String;
        final bytes = (await rootBundle.load('assets/phrase_bank/$file')).buffer.asUint8List();
        out[e.key] = wavPcm(bytes);
      }
      return PhraseClips(out);
    } catch (_) {
      return PhraseClips(const {});
    }
  }

  /// Concatenates the clips for [parts] with [gapMs] of silence between them. Unknown keys are skipped.
  /// Returns null if nothing could be built.
  Uint8List? build(List<String> parts, {int gapMs = 70}) {
    final clips = [for (final p in parts) if (_pcm[p] != null) _pcm[p]!];
    if (clips.isEmpty) return null;
    final gap = gapMs * kClipSampleRate ~/ 1000 * 2;
    final total = clips.fold<int>(0, (a, c) => a + c.length) + gap * (clips.length - 1);
    final out = Uint8List(total); // silence is zeros
    var o = 0;
    for (var i = 0; i < clips.length; i++) {
      out.setRange(o, o + clips[i].length, clips[i]);
      o += clips[i].length;
      if (i < clips.length - 1) o += gap;
    }
    return out;
  }
}

/// Plays utterances from the phrase bank on the 'guidance' voice.
class ClipSpeechOutput implements SpeechOutput {
  ClipSpeechOutput(this._clips, this._voice);

  final PhraseClips _clips;
  final VoiceOut _voice;

  @override
  Future<void> speak(List<String> parts) async {
    final pcm = _clips.build(parts);
    if (pcm == null) return;
    await _voice.enqueue('guidance', pcm, wait: true);
  }

  @override
  Future<void> stop() => _voice.flush('guidance');
}
