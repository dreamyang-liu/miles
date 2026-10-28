"""The qwen3 mask generator must reproduce the template's own rendering.

Two things real chat templates do that per-message rendering has to account
for: they prepend an implicit system preamble when the conversation opens
without a system message, and they fold consecutive tool results into one
turn. A synthetic word-level tokenizer makes both visible without a checkpoint.
"""

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from miles.utils.mask_utils import MultiTurnLossMaskGenerator

WORDS = [
    "<unk>", "<s>", "</s>", "<u>", "<a>", "<t>", "</m>", "<think>", "FOR", "TESTING", "CALCULATING", "LOSS", "MASK",
    "ONLY", "preamble", "task", "plan", "look", "run", "ls", "cat", "out1", "out2", "done", "sys", "<x>",
]

# ``<s> preamble </m>`` opens a conversation that has no system message; consecutive
# tool results share one ``<t> ... </m>`` turn; an assistant turn starts with a
# generation prompt ``<a> <think>`` so the mask has something to skip.
TEMPLATE = (
    "{% if messages[0].role != 'system' %}<s> preamble </m> {% endif %}"
    "{% for m in messages %}"
    "{% if m.role == 'tool' %}"
    "{% if loop.first or loop.previtem.role != 'tool' %}<t> {% endif %}"
    "{{ m.content }} "
    "{% if loop.last or loop.nextitem.role != 'tool' %}</m> {% endif %}"
    "{% elif m.role == 'assistant' %}<a> <think> {{ m.content }} </m> "
    "{% else %}{{ {'user':'<u>', 'system':'<s>'}[m.role] }} {{ m.content }} </m> {% endif %}"
    "{% endfor %}{% if add_generation_prompt %}<a> <think> {% endif %}"
)


@pytest.fixture
def tokenizer():
    raw = Tokenizer(models.WordLevel({word: i for i, word in enumerate(WORDS)}, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    result = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>")
    result.chat_template = TEMPLATE
    return result


CONVERSATION = [
    {"role": "user", "content": "task"},
    {"role": "assistant", "content": "plan run ls cat"},
    {"role": "tool", "content": "out1"},
    {"role": "tool", "content": "out2"},
    {"role": "assistant", "content": "look done"},
]


def trained(tokenizer, ids, mask):
    return tokenizer.decode([t for t, keep in zip(ids, mask, strict=True) if keep])


@pytest.mark.parametrize("with_system", [False, True])
def test_qwen3_tokens_equal_the_templates_own_rendering(tokenizer, with_system):
    messages = ([{"role": "system", "content": "sys"}] if with_system else []) + CONVERSATION
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")
    assert generator.system_message_length > 0, "the implicit preamble is what this test is about"
    ids, mask = generator.get_loss_mask(messages)
    assert ids == tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
    assert len(mask) == len(ids)
    text = trained(tokenizer, ids, mask)
    assert text.split() == ["plan", "run", "ls", "cat", "</m>", "look", "done", "</m>"]
    assert "<think>" not in text, "the generation prompt is not a target"
    assert "out1" not in text and "preamble" not in text


def test_consecutive_tool_results_are_one_untrained_turn(tokenizer):
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")
    ids, mask = generator.get_loss_mask(CONVERSATION)
    rendered = tokenizer.decode(ids)
    assert rendered.count("<t>") == 1, "two tool results, one tool turn, as the template renders them"
    tool_turn = tokenizer.decode(ids)[rendered.index("<t>"):]
    assert tool_turn.startswith("<t> out1 out2 </m>")


def test_step_loss_mask_still_silences_a_message(tokenizer):
    messages = [dict(message) for message in CONVERSATION]
    messages[1]["step_loss_mask"] = 0
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")
    ids, mask = generator.get_loss_mask(messages)
    assert ids == tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
    assert trained(tokenizer, ids, mask).split() == ["look", "done", "</m>"]


def test_a_template_that_rewrites_the_prefix_is_refused_not_misaligned(tokenizer):
    """A template whose opening depends on the conversation length renders the
    probe message differently alone and before another message, so no slice of
    one rendering is the other: the generator must say so, not strip blindly."""
    tokenizer.chat_template = (
        "{% if messages|length > 1 %}<x> {% endif %}{% for m in messages %}"
        "{{ {'user':'<u>', 'assistant':'<a>', 'tool':'<t>', 'system':'<s>'}[m.role] }} {{ m.content }} </m> {% endfor %}"
    )
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")
    with pytest.raises(ValueError, match="does not preserve"):
        generator.get_loss_mask(CONVERSATION[:2])
