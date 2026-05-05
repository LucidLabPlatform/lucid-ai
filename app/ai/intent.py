"""Two-tier intent classifier for routing user messages to specialist agents.

Tier 1 (fast): Regex keyword matching — covers ~90% of queries with no LLM call.
Tier 2 (slow): Single-token LLM classification — fallback for ambiguous queries.
"""

import logging
import re

log = logging.getLogger(__name__)

INTENTS = ("fleet", "command", "experiment", "topic_link", "logs", "conversation")

# Each pattern list is checked against the lowercased user message.
# Order matters: more specific patterns should come first within each group.
_INTENT_PATTERNS: dict[str, list[re.Pattern]] = {
    "fleet": [
        re.compile(r"\b(what|which|show|list|how many)\b.*\b(agents?|devices?|components?)\b", re.I),
        re.compile(r"\b(agents?|devices?)\b.*\b(online|offline|status|registered|connected)\b", re.I),
        re.compile(r"\b(fleet|sync)\s*(state|status|summary)\b", re.I),
        re.compile(r"\bare\s+(any|all|there)\b.*\b(agents?|devices?)\b", re.I),
        re.compile(r"\bwhat\s+is\s+(the\s+)?(state|status)\s+of\b", re.I),
        re.compile(r"\bfull\s+state\b", re.I),
    ],
    "command": [
        re.compile(r"\b(set|turn|switch|change|toggle)\b.*\b(led|light|color|colour|brightness|strip)\b", re.I),
        re.compile(r"\b(send|execute|fire|dispatch)\b.*\b(command|cmd)\b", re.I),
        re.compile(r"\b(get|show|what)\b.*\b(command\s*catalog|available\s*commands?)\b", re.I),
        re.compile(r"\bwhat\s+commands?\s+(can|do|are|could)\b", re.I),
        re.compile(r"\b(restart|reboot|shutdown|ping|configure)\b.*\b(agents?|devices?|components?)\b", re.I),
        re.compile(r"\bping\s+(all|every|each)\b", re.I),
        re.compile(r"\b(set|turn)\s+(the\s+)?(led|light|strip)\b", re.I),
        re.compile(r"\bturn\s+(on|off)\b", re.I),
        re.compile(r"\b(batch|all)\b.*\b(command|set|turn|ping)\b", re.I),
        re.compile(r"\b(set|change)\s+.*\s+to\s+(red|green|blue|yellow|white|purple|orange|pink)\b", re.I),
    ],
    "experiment": [
        re.compile(r"\b(experiment|trial)\b", re.I),
        re.compile(r"\b(template|foraging|calibration)\b.*\b(experiment|run|start)\b", re.I),
        re.compile(r"\b(start|run|launch|begin|cancel|approve|stop)\b.*\b(experiment|trial)\b", re.I),
        re.compile(r"\b(experiment|run)\s*(status|result|step|progress)\b", re.I),
        re.compile(r"\b(status|state)\b.*\b(experiment|trial|run)\b", re.I),
        re.compile(r"\blist\b.*\b(templates?|experiments?|runs?)\b", re.I),
        re.compile(r"\bapprove\b.*\bstep\b", re.I),
        re.compile(r"\blast\s+(experiment|run|trial)\b", re.I),
    ],
    "topic_link": [
        re.compile(r"\btopic\s*links?\b", re.I),
        re.compile(r"\b(link|bridge|forward|route|republish)\b.*\b(topic|mqtt|perception|sensor)\b", re.I),
        re.compile(r"\b(create|activate|deactivate|delete|show|list)\b.*\blinks?\b", re.I),
        re.compile(r"\bthroughput\b", re.I),
        re.compile(r"\b(emqx|rule|routing)\b.*\b(rule|link|message)\b", re.I),
        re.compile(r"\bmessage\s*(count|rate|forwarding)\b", re.I),
    ],
    "logs": [
        re.compile(r"\b(show|get|view|recent|latest)\b.*\blogs?\b", re.I),
        re.compile(r"\b(command|cmd)\s*(history|log)\b", re.I),
        re.compile(r"\bwhat\s+(commands?|was)\s+(were\s+)?(sent|executed|run)\b", re.I),
        re.compile(r"\bcommands?\s+(sent|were|executed|run|issued)\b", re.I),
        re.compile(r"\blogs?\s+from\b", re.I),
        re.compile(r"\brecent\s+(commands?|logs?)\b", re.I),
    ],
}

# Minimum confidence to accept keyword classification without LLM fallback.
_CONFIDENCE_THRESHOLD = 0.7

# Minimum separation between top two intents to accept without fallback.
_SEPARATION_THRESHOLD = 0.3


def classify_by_keywords(message: str) -> tuple[str, float]:
    """Score each intent by counting regex pattern matches.

    Returns:
        Tuple of (intent, confidence).
        Confidence is 0.0-1.0 based on match count relative to other intents.
        Returns ("conversation", 0.0) if no patterns match.
    """
    scores: dict[str, int] = {}
    for intent, patterns in _INTENT_PATTERNS.items():
        score = sum(1 for p in patterns if p.search(message))
        if score > 0:
            scores[intent] = score

    if not scores:
        return "conversation", 0.0

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_intent, top_score = ranked[0]
    total = sum(s for _, s in ranked)
    confidence = top_score / total if total > 0 else 0.0

    # Check separation from second-best
    if len(ranked) > 1:
        second_score = ranked[1][1]
        separation = (top_score - second_score) / total
        if separation < _SEPARATION_THRESHOLD:
            # Two intents are too close — lower confidence to trigger LLM fallback
            confidence = min(confidence, _CONFIDENCE_THRESHOLD - 0.1)

    return top_intent, confidence


async def classify_intent(
    message: str,
    llm=None,
    classify_prompt: str = "",
) -> tuple[str, float, str]:
    """Two-tier intent classification.

    Args:
        message:          User's message text.
        llm:              ChatOllama instance for LLM fallback (optional).
        classify_prompt:  System prompt for the LLM classifier.

    Returns:
        Tuple of (intent, confidence, method) where method is "keyword" or "llm".
    """
    intent, confidence = classify_by_keywords(message)

    if confidence >= _CONFIDENCE_THRESHOLD:
        log.info(
            "Intent classified by keywords: %s (confidence=%.2f)",
            intent, confidence,
        )
        return intent, confidence, "keyword"

    # Tier 2: LLM fallback
    if llm is None:
        log.warning("LLM not available for intent fallback, using keyword result: %s", intent)
        return intent, confidence, "keyword"

    try:
        response = await llm.ainvoke([
            ("system", classify_prompt),
            ("user", message),
        ])
        llm_intent = response.content.strip().lower().replace(" ", "_")

        # Validate the LLM returned a known intent
        if llm_intent in INTENTS:
            log.info("Intent classified by LLM: %s (keyword was: %s)", llm_intent, intent)
            return llm_intent, 0.8, "llm"

        log.warning(
            "LLM returned unknown intent '%s', falling back to keyword: %s",
            llm_intent, intent,
        )
    except Exception as e:
        log.warning("LLM intent classification failed: %s, using keyword: %s", e, intent)

    return intent, confidence, "keyword"
