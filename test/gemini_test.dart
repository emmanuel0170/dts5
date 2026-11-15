import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:spatial_nav/audio/phrase_clips.dart';
import 'package:spatial_nav/decision/decision_engine.dart';
import 'package:spatial_nav/decision/guidance.dart';
import 'package:spatial_nav/decision/phrase_bank.dart';
import 'package:spatial_nav/decision/speech_arbiter.dart';
import 'package:spatial_nav/gemini/assistant_tools.dart';
import 'package:spatial_nav/gemini/live_protocol.dart';

class _Silent implements SpeechOutput {
  @override
  Future<void> speak(List<String> parts) async {}
  @override
  Future<void> stop() async {}
}

/// Minimal canonical 44-byte-header WAV: 16-bit mono PCM.
Uint8List wav(List<int> samples, {int rate = 24000, int channels = 1}) {
  final data = ByteData(samples.length * 2);
  for (var i = 0; i < samples.length; i++) {
    data.setInt16(i * 2, samples[i], Endian.little);
  }
  final h = ByteData(44);
  void tag(int o, String s) {
    for (var i = 0; i < 4; i++) {
      h.setUint8(o + i, s.codeUnitAt(i));
    }
  }

  tag(0, 'RIFF');
  h.setUint32(4, 36 + data.lengthInBytes, Endian.little);
  tag(8, 'WAVE');
  tag(12, 'fmt ');
  h.setUint32(16, 16, Endian.little);
  h.setUint16(20, 1, Endian.little);
  h.setUint16(22, channels, Endian.little);
  h.setUint32(24, rate, Endian.little);
  h.setUint32(28, rate * channels * 2, Endian.little);
  h.setUint16(32, channels * 2, Endian.little);
  h.setUint16(34, 16, Endian.little);
  tag(36, 'data');
  h.setUint32(40, data.lengthInBytes, Endian.little);
  return Uint8List.fromList([...h.buffer.asUint8List(), ...data.buffer.asUint8List()]);
}

void main() {
  group('server messages', () {
    test('audio part, transcripts and turnComplete in one message', () {
      final pcm = Uint8List.fromList([1, 2, 3, 4]);
      final events = parseServerMessage(jsonEncode({
        'serverContent': {
          'modelTurn': {
            'parts': [
              {
                'inlineData': {'mimeType': 'audio/pcm;rate=24000', 'data': base64Encode(pcm)},
              },
            ],
          },
          'outputTranscription': {'text': 'Turn left.'},
          'turnComplete': true,
        },
      }));
      expect(events[0], isA<AudioChunk>());
      expect((events[0] as AudioChunk).pcm, pcm);
      expect((events[1] as OutputTranscript).text, 'Turn left.');
      expect(events[2], isA<TurnComplete>());
    });

    test('binary frames are accepted', () {
      final raw = utf8.encode(jsonEncode({'setupComplete': {}}));
      expect(parseServerMessage(raw).single, isA<SetupComplete>());
    });

    test('interrupted comes before audio in the same message', () {
      final events = parseServerMessage(jsonEncode({
        'serverContent': {
          'interrupted': true,
          'modelTurn': {
            'parts': [
              {
                'inlineData': {'mimeType': 'audio/pcm', 'data': base64Encode([0, 0])},
              },
            ],
          },
        },
      }));
      expect(events.first, isA<Interrupted>());
      expect(events.last, isA<AudioChunk>());
    });

    test('tool call, goAway and resumption update', () {
      final tool = parseServerMessage(jsonEncode({
        'toolCall': {
          'functionCalls': [
            {
              'id': 'c1',
              'name': 'set_search_target',
              'args': {
                'labels': ['chair'],
              },
            },
          ],
        },
      })).single as ToolCall;
      expect(tool.calls.single.name, 'set_search_target');
      expect(tool.calls.single.args['labels'], ['chair']);

      final away = parseServerMessage(jsonEncode({
        'goAway': {'timeLeft': '29.5s'},
      })).single as GoAway;
      expect(away.timeLeft, const Duration(milliseconds: 29500));

      final up = parseServerMessage(jsonEncode({
        'sessionResumptionUpdate': {'newHandle': 'h1', 'resumable': true},
      })).single as ResumptionUpdate;
      expect(up.handle, 'h1');
      expect(up.resumable, isTrue);
    });
  });

  group('client messages', () {
    test('setup carries model, audio modality, tools, resumption and transcription', () {
      final m = buildSetup(
        model: 'gemini-3.8-live',
        systemInstruction: 'sys',
        functionDeclarations: toolDeclarations(['chair', 'door']),
        resumptionHandle: 'h1',
      )['setup'] as Map<String, dynamic>;
      expect(m['model'], 'models/gemini-3.8-live');
      expect((m['generationConfig'] as Map)['responseModalities'], ['AUDIO']);
      expect(m.containsKey('inputAudioTranscription'), isTrue);
      expect(m.containsKey('outputAudioTranscription'), isTrue);
      expect((m['sessionResumption'] as Map)['handle'], 'h1');
      final decls = ((m['tools'] as List).single as Map)['functionDeclarations'] as List;
      expect(decls.map((d) => (d as Map)['name']), ['set_search_target', 'cancel_search', 'get_scene', 'look_around']);
      final enumValues =
          (((decls.first as Map)['parameters'] as Map)['properties'] as Map)['labels']['items']['enum'] as List;
      expect(enumValues, ['chair', 'door']);
    });

    test('audio and snapshot turn', () {
      final a = msgAudio(Uint8List.fromList([1, 2]))['realtimeInput'] as Map;
      expect((a['audio'] as Map)['mimeType'], 'audio/pcm;rate=16000');
      final turn = msgClientTurn(text: 'look', jpeg: Uint8List.fromList([9]))['clientContent'] as Map;
      expect(turn['turnComplete'], isTrue);
      final parts = ((turn['turns'] as List).single as Map)['parts'] as List;
      expect((parts.first as Map).containsKey('inlineData'), isTrue);
      expect((parts.last as Map)['text'], 'look');
    });

    test('tool response matches calls by id', () {
      final r = msgToolResponse([const ToolReply('c1', 'get_scene', {'ok': true})])['toolResponse'] as Map;
      expect(((r['functionResponses'] as List).single as Map)['id'], 'c1');
    });
  });

  group('assistant tools', () {
    late DecisionEngine decision;
    late AssistantTools tools;
    var viewsSent = 0;

    setUp(() {
      decision = DecisionEngine(speech: SpeechArbiter(_Silent()));
      viewsSent = 0;
      tools = AssistantTools(
        decision: decision,
        vocabulary: ['chair', 'Traffic Light', 'door'],
        sendCurrentView: () async => viewsSent++,
      );
    });

    test('set_search_target maps words onto the vocabulary and starts the search', () async {
      final r = await tools.handle(const FunctionCall('1', 'set_search_target', {
        'labels': ['CHAIR', 'sofa'],
        'spoken_name': 'a chair',
      }));
      expect(r.response['ok'], isTrue);
      expect(r.response['searching_for'], ['chair']); // canonical spelling; 'sofa' is not in this vocabulary
      expect(decision.guidance.goal?.labels, ['chair']);
      expect(decision.guidance.phase, Phase.scanning);
    });

    test('onSearchStarted fires for a successful search only', () async {
      var started = 0;
      final tools2 = AssistantTools(
        decision: decision,
        vocabulary: ['chair'],
        sendCurrentView: () async {},
        onSearchStarted: () => started++,
      );
      await tools2.handle(const FunctionCall('1', 'set_search_target', {
        'labels': ['keys'],
      }));
      expect(started, 0);
      await tools2.handle(const FunctionCall('2', 'set_search_target', {
        'labels': ['chair'],
      }));
      expect(started, 1);
    });

    test('words outside the vocabulary are refused with an explanation', () async {
      final r = await tools.handle(const FunctionCall('2', 'set_search_target', {
        'labels': ['keys'],
      }));
      expect(r.response['ok'], isFalse);
      expect(decision.guidance.goal, isNull);
    });

    test('cancel_search, get_scene and look_around', () async {
      await tools.handle(const FunctionCall('1', 'set_search_target', {
        'labels': ['door'],
      }));
      final scene = await tools.handle(const FunctionCall('3', 'get_scene', {}));
      expect(scene.response['search_active'], isTrue);
      expect(scene.response['searching_for'], ['door']);
      await tools.handle(const FunctionCall('4', 'look_around', {}));
      expect(viewsSent, 1);
      await tools.handle(const FunctionCall('5', 'cancel_search', {}));
      expect(decision.guidance.goal, isNull);
    });

    test('unknown tool', () async {
      final r = await tools.handle(const FunctionCall('6', 'dance', {}));
      expect(r.response['ok'], isFalse);
    });
  });

  group('phrase clips', () {
    test('wavPcm extracts the data chunk and validates the format', () {
      final pcm = wavPcm(wav([1, -1, 2]));
      expect(pcm.length, 6);
      expect(() => wavPcm(wav([1], rate: 22050)), throwsFormatException);
      expect(() => wavPcm(wav([1], channels: 2)), throwsFormatException);
      expect(() => wavPcm(Uint8List(10)), throwsFormatException);
    });

    test('build concatenates clips with a gap and skips unknown keys', () {
      final a = wavPcm(wav(List.filled(100, 1000)));
      final b = wavPcm(wav(List.filled(50, -1000)));
      final clips = PhraseClips({'say:found': a, 'label:chair': b});
      final out = clips.build(['say:found', 'nope', 'label:chair'], gapMs: 10)!;
      const gapBytes = 10 * 24000 ~/ 1000 * 2;
      expect(out.length, a.length + gapBytes + b.length);
      expect(out.sublist(a.length, a.length + gapBytes).every((v) => v == 0), isTrue);
      expect(clips.build(['nope']), isNull);
    });
  });
}
