import '../decision/decision_engine.dart';
import 'live_protocol.dart';

String systemInstruction(List<String> vocabulary) => '''
You are the voice of a handheld navigation aid. The user may be blind or have low vision and is probably walking, so keep every reply to one or two short sentences of plain spoken language.

The device runs its own object detector and depth model, and it guides the user with sounds and short spoken phrases. While a search is active, do not narrate directions or distances yourself.

To find something, call set_search_target with the closest matching words from the detector vocabulary below. If nothing in the vocabulary fits, say so plainly and offer to describe what the camera sees instead. Call cancel_search when the user wants to stop.
Call get_scene to learn what the detector currently sees, with distances. Call look_around to receive a fresh camera image before you describe the surroundings.

Messages that begin with [scene-check] come from the device, not the user: they carry a camera image because the local detector is struggling. Answer briefly: say whether you can see what the user is looking for, and which way they should turn. Never give distances from an image.

Safety: never tell the user that a path is clear or safe to walk, and never that it is safe to cross a road. If you are unsure, say that you are not sure.

Detector vocabulary: ${vocabulary.join(', ')}.
''';

List<Map<String, dynamic>> toolDeclarations(List<String> vocabulary) => [
      {
        'name': 'set_search_target',
        'description': 'Start guiding the user toward an object. Pass the closest matching words from the detector '
            'vocabulary (synonyms are fine, e.g. both "couch" and "sofa" if both are in the vocabulary).',
        'behavior': 'BLOCKING',
        'parameters': {
          'type': 'OBJECT',
          'properties': {
            'labels': {
              'type': 'ARRAY',
              'description': 'One or more detector vocabulary words that would count as the object.',
              'items': {'type': 'STRING', 'enum': vocabulary},
            },
            'spoken_name': {'type': 'STRING', 'description': 'What the user called it, in their words.'},
          },
          'required': ['labels'],
        },
      },
      {
        'name': 'cancel_search',
        'description': 'Stop the current search and the guidance sounds.',
        'behavior': 'BLOCKING',
      },
      {
        'name': 'get_scene',
        'description': 'What the on-device detector currently sees: objects with direction and distance, plus the '
            'state of the current search.',
        'behavior': 'BLOCKING',
      },
      {
        'name': 'look_around',
        'description': 'Send a fresh camera image to you so that you can describe the surroundings.',
        'behavior': 'BLOCKING',
      },
    ];

/// Executes the model's tool calls against the decision engine. Every tool is instant and idempotent, so
/// cancellations from the server can be ignored.
class AssistantTools {
  AssistantTools({
    required this.decision,
    required List<String> vocabulary,
    required this.sendCurrentView,
    this.onSearchStarted,
  }) : _canonical = {for (final v in vocabulary) v.toLowerCase(): v};

  final DecisionEngine decision;

  /// Sends a fresh camera frame to Gemini.
  final Future<void> Function() sendCurrentView;

  /// Called when set_search_target succeeded: the device now guides, so the conversation can wind down.
  final void Function()? onSearchStarted;
  final Map<String, String> _canonical;

  Future<ToolReply> handle(FunctionCall call) async {
    Map<String, dynamic> reply(Map<String, dynamic> r) => r;
    switch (call.name) {
      case 'set_search_target':
        final asked = [for (final l in (call.args['labels'] as List? ?? const [])) l.toString()];
        final known = [for (final l in asked) if (_canonical[l.toLowerCase()] != null) _canonical[l.toLowerCase()]!];
        if (known.isEmpty) {
          return ToolReply(call.id, call.name, reply({
            'ok': false,
            'error': 'None of those words are in the detector vocabulary, so the device cannot track them live. '
                'Tell the user, and offer to describe the scene with look_around.',
          }));
        }
        decision.setGoal(known, (call.args['spoken_name'] as String?) ?? known.first);
        onSearchStarted?.call();
        return ToolReply(call.id, call.name, reply({
          'ok': true,
          'searching_for': known,
          'note': 'The device now guides the user with sound and short phrases. Keep your own reply very short.',
        }));
      case 'cancel_search':
        decision.clearGoal();
        return ToolReply(call.id, call.name, reply({'ok': true}));
      case 'get_scene':
        final g = decision.guidance;
        return ToolReply(call.id, call.name, reply({
          'scene': decision.sceneSummary(),
          'search_active': g.goal != null,
          'search_phase': g.phase.name,
          'searching_for': g.goal?.labels ?? const <String>[],
        }));
      case 'look_around':
        await sendCurrentView();
        return ToolReply(call.id, call.name, reply({'ok': true, 'note': 'A fresh camera image was just sent.'}));
      default:
        return ToolReply(call.id, call.name, reply({'ok': false, 'error': 'unknown tool'}));
    }
  }
}
