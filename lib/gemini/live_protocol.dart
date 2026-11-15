import 'dart:convert';
import 'dart:typed_data';

// Message shapes follow the Live API WebSockets reference (Gemini API, v1beta, updated 2026-10).

/// Server -> client events, the subset this app uses.
sealed class LiveEvent {
  const LiveEvent();
}

class SetupComplete extends LiveEvent {
  const SetupComplete();
}

class AudioChunk extends LiveEvent {
  const AudioChunk(this.pcm, this.mimeType);
  final Uint8List pcm; // 16-bit little-endian mono, 24 kHz for the Live models
  final String mimeType;
}

class ModelText extends LiveEvent {
  const ModelText(this.text);
  final String text;
}

class InputTranscript extends LiveEvent {
  const InputTranscript(this.text);
  final String text;
}

class OutputTranscript extends LiveEvent {
  const OutputTranscript(this.text);
  final String text;
}

class TurnComplete extends LiveEvent {
  const TurnComplete();
}

/// The model's turn was cut off (the user spoke over it): stop and clear any queued playback.
class Interrupted extends LiveEvent {
  const Interrupted();
}

class FunctionCall {
  const FunctionCall(this.id, this.name, this.args);
  final String id;
  final String name;
  final Map<String, dynamic> args;
}

class ToolCall extends LiveEvent {
  const ToolCall(this.calls);
  final List<FunctionCall> calls;
}

class ToolCallCancellation extends LiveEvent {
  const ToolCallCancellation(this.ids);
  final List<String> ids;
}

class GoAway extends LiveEvent {
  const GoAway(this.timeLeft);
  final Duration? timeLeft;
}

class ResumptionUpdate extends LiveEvent {
  const ResumptionUpdate(this.handle, this.resumable);
  final String? handle;
  final bool resumable;
}

/// Not from the server: the socket ended.
class SocketClosed extends LiveEvent {
  const SocketClosed([this.code, this.reason]);
  final int? code;
  final String? reason;
}

/// One server message can carry several things (audio parts, a transcript, turnComplete), so this returns a list.
/// The server may send JSON in text or binary frames; both are accepted.
List<LiveEvent> parseServerMessage(Object raw) {
  final text = raw is String ? raw : utf8.decode(raw as List<int>);
  final m = jsonDecode(text) as Map<String, dynamic>;
  final out = <LiveEvent>[];

  if (m.containsKey('setupComplete')) out.add(const SetupComplete());

  final sc = m['serverContent'];
  if (sc is Map<String, dynamic>) {
    if (sc['interrupted'] == true) out.add(const Interrupted()); // before any audio in the same message
    final parts = ((sc['modelTurn'] as Map?)?['parts'] as List?) ?? const [];
    for (final p in parts) {
      final pm = p as Map<String, dynamic>;
      final inline = pm['inlineData'];
      if (inline is Map && inline['data'] is String) {
        out.add(AudioChunk(base64Decode(inline['data'] as String), (inline['mimeType'] as String?) ?? ''));
      } else if (pm['text'] is String) {
        out.add(ModelText(pm['text'] as String));
      }
    }
    final it = sc['inputTranscription'];
    if (it is Map && it['text'] is String) out.add(InputTranscript(it['text'] as String));
    final ot = sc['outputTranscription'];
    if (ot is Map && ot['text'] is String) out.add(OutputTranscript(ot['text'] as String));
    if (sc['turnComplete'] == true) out.add(const TurnComplete());
  }

  final tc = m['toolCall'];
  if (tc is Map) {
    out.add(ToolCall([
      for (final c in (tc['functionCalls'] as List? ?? const []))
        FunctionCall(
          (c as Map)['id'] as String? ?? '',
          c['name'] as String? ?? '',
          Map<String, dynamic>.from((c['args'] as Map?) ?? const {}),
        ),
    ]));
  }
  final tcc = m['toolCallCancellation'];
  if (tcc is Map) out.add(ToolCallCancellation([for (final i in (tcc['ids'] as List? ?? const [])) i as String]));

  final ga = m['goAway'];
  if (ga is Map) out.add(GoAway(_parseDuration(ga['timeLeft'])));

  final sru = m['sessionResumptionUpdate'];
  if (sru is Map) out.add(ResumptionUpdate(sru['newHandle'] as String?, sru['resumable'] == true));

  return out;
}

/// Protobuf JSON durations look like "30s" or "29.5s".
Duration? _parseDuration(Object? v) {
  if (v is! String || !v.endsWith('s')) return null;
  final secs = double.tryParse(v.substring(0, v.length - 1));
  return secs == null ? null : Duration(milliseconds: (secs * 1000).round());
}

// ---------------------------------------------------------------- client messages

Map<String, dynamic> buildSetup({
  required String model,
  required String systemInstruction,
  required List<Map<String, dynamic>> functionDeclarations,
  String? voiceName,
  String? resumptionHandle,
}) =>
    {
      'setup': {
        'model': 'models/$model',
        'generationConfig': {
          'responseModalities': ['AUDIO'], // the only supported response modality; text comes as transcription
          if (voiceName != null)
            'speechConfig': {
              'voiceConfig': {
                'prebuiltVoiceConfig': {'voiceName': voiceName},
              },
            },
        },
        'systemInstruction': {
          'parts': [
            {'text': systemInstruction},
          ],
        },
        'tools': [
          {'functionDeclarations': functionDeclarations},
        ],
        'sessionResumption': {if (resumptionHandle != null) 'handle': resumptionHandle},
        'contextWindowCompression': {'slidingWindow': {}},
        'inputAudioTranscription': {},
        'outputAudioTranscription': {},
      },
    };

Map<String, dynamic> msgAudio(Uint8List pcm16k) => {
      'realtimeInput': {
        'audio': {'data': base64Encode(pcm16k), 'mimeType': 'audio/pcm;rate=16000'},
      },
    };

Map<String, dynamic> msgAudioStreamEnd() => {
      'realtimeInput': {'audioStreamEnd': true},
    };

/// A camera frame as live video input. Turn coverage defaults to "all video", so send frames only when needed.
Map<String, dynamic> msgVideoJpeg(Uint8List jpeg) => {
      'realtimeInput': {
        'video': {'data': base64Encode(jpeg), 'mimeType': 'image/jpeg'},
      },
    };

Map<String, dynamic> msgText(String text) => {
      'realtimeInput': {'text': text},
    };

/// One explicit user turn with an image and a prompt. turnComplete interrupts any generation in progress.
Map<String, dynamic> msgClientTurn({required String text, Uint8List? jpeg}) => {
      'clientContent': {
        'turns': [
          {
            'role': 'user',
            'parts': [
              if (jpeg != null)
                {
                  'inlineData': {'mimeType': 'image/jpeg', 'data': base64Encode(jpeg)},
                },
              {'text': text},
            ],
          },
        ],
        'turnComplete': true,
      },
    };

class ToolReply {
  const ToolReply(this.id, this.name, this.response);
  final String id;
  final String name;
  final Map<String, dynamic> response;
}

Map<String, dynamic> msgToolResponse(List<ToolReply> replies) => {
      'toolResponse': {
        'functionResponses': [
          for (final r in replies) {'id': r.id, 'name': r.name, 'response': r.response},
        ],
      },
    };
