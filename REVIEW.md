# Blog Series Review: LLM Audit & Guard Posts

Reviewed: 2026-06-02
Scope: All posts under `content/posts/` except `dual-index-rag/`
Perspective: External reader evaluating the blog for technical credibility and audience appeal

---

## Series Overview

Seven posts covering LLM runtime security infrastructure:

| Post | Slug | Role in series |
|------|------|----------------|
| Part 1: Zero-Touch LLM Guarding | `zero-touch-llm-guarding` | Architecture: callback injection, context gating, opt-in activation |
| Part 2: Prompt Injection Patterns | `llm-guard-patterns` | Detection logic: regex patterns, base64 decoding, structural isolation, output scanning |
| Part 3: ML-Based Injection Detection | `llm-guard-classifiers` | Classifier evaluation: Prompt Guard 2, online/offline deployment, fine-tuning, red teaming |
| Part 4: LangChain Middleware Guards | `llm-guard-middleware` | Translation guide: same guards via middleware instead of callbacks |
| Healthcare LLM Security | `llm-security-healthcare` | Domain application: how generic security assumptions break in healthcare |
| Template Method for ML | `template-method-ml` | Design pattern: why template method fits ML infra lifecycle enforcement |
| Zero-Touch Audit Logging | `zero-touch-llm-auditing` | Audit framework: automatic prompt/response capture, invocation correlation, two-tier storage |

All posts are `draft: true`. None are published.

---

## Strategic Direction from Author

- The author does **not** want to be positioned as a healthcare AI specialist. The blog targets a broad ML/AI engineering audience.
- This series covers audit/guard work only. Other series (not yet written) will cover different ML/AI topics.
- The goal is to demonstrate production-grade LLM infrastructure skills to a general technical audience.

This has direct implications for the healthcare post and for how domain-specific examples are used throughout the series.

---

## What Works Well

### Strong central insight (Part 2)
The intent-vs-keyword distinction is the series' best idea. "Ignore previous medications" vs "ignore previous instructions" — the discriminator is the object of the verb, not the verb itself. This is memorable, well-illustrated with comparison tables, and genuinely useful. Protect this in any restructuring.

### Solid architecture (Parts 1, audit post)
The three-layer design (callback at factory, contextvars scope, base class activation) is well-reasoned. The opt-in gating via `GuardContext` — callbacks fire on every model but do nothing without an active context — is a clean engineering decision. The invocation chain correlation in the audit post (why per-call logging doesn't solve multi-step requests) is strong technical content.

### Healthcare post writing quality
The 6-problem structure is effective. Each section follows a clear arc: assumption that breaks, concrete example, what works instead. The "safety filters are backwards" section (Problem 3) is the strongest — it's a genuine inversion that most readers won't have considered. The writing here has more personality and confidence than the other posts.

### Code examples
Throughout the series, code examples are well-scoped — showing the minimum needed to make the point, with callouts to the 2-3 details that matter. The pattern of showing a code block followed by "Three details that matter:" is effective.

### References and citations
Properly cited with inline references linking to numbered entries. OWASP, CrowdStrike, Meta, Aptible sources are credible and current.

---

## Structural Problems

### 1. Too many posts for one body of work

Seven posts is too many for a single series about one system. The cross-linking is dense — every post references 2-3 others, and several can't stand alone. This creates a high barrier for a blog reader who expects to get value from a single post.

**Recommendation:** Consolidate to 4-5 posts:
- Merge Parts 1 and 2 into a single post. The architecture (callbacks, context gating) and the detection logic (regex patterns, base64, structural isolation, output scanning) are one idea. Splitting them forces the reader to bounce between posts for a complete picture.
- Fold Part 4 (middleware) into Part 1 as a section ("If you use create_agent()") or cut it entirely. The post acknowledges it's a "translation guide" — at ~200 lines, it's too thin to justify a standalone article.
- The remaining posts (classifiers, healthcare, template method, audit) can each stand alone with minor adjustments.

### 2. Repetition across posts

The same concepts are re-explained from scratch in multiple posts:

- The "ignore previous medications" vs "ignore previous instructions" example appears in Parts 1, 2, and the healthcare post.
- The `create_model()` factory function and callback injection is explained in Part 1, the audit post, and partially in Part 4.
- The `GuardedAgent` / `AuditedAgent` inheritance pattern appears in Parts 1, the template method post, and the audit post.
- The `guard_context` / `audit_scope` context manager pattern is explained twice with near-identical code.

**Recommendation:** Each concept should have one canonical explanation in one post. Other posts should reference it with a one-sentence summary and a link, not re-explain. This also shortens the posts.

### 3. No production outcomes

This is the single highest-leverage gap. Every post describes what was designed and built, but none describe what happened when it ran. Questions a reader will have:

- How many injection attempts did the regex patterns catch?
- What was the false positive rate on domain-specific language?
- How many findings per day does the system produce?
- What did the offline 86M classifier find that the online 22M missed?
- How much did the base64 scanning layer actually contribute?
- What did red teaming reveal?

**Recommendation:** Add one section per post with rough production outcomes. Even order-of-magnitude numbers ("the regex layer flagged ~200 inputs per week, ~15% were true positives, the rest were clinical language that led us to refine 3 patterns") would transform the series from architecture documentation into engineering case study. This is the difference between "I designed this" and "I shipped this and learned from it."

### 4. Healthcare framing conflicts with author's positioning goal

The healthcare post is the best-written piece in the series, but if it's prominent, readers will mentally categorize the author as "healthcare AI." The author explicitly wants to target a broader market.

**Recommendation:**
- Reposition the healthcare post as an application/adaptation post at the end of the series, not a centerpiece. Frame it as: "Here's how the framework adapts to a regulated domain."
- Replace healthcare-specific examples in other posts with generic ones. The intent-vs-keyword insight works with any domain that has specialized vocabulary — legal ("disregard the earlier testimony"), customer support ("override the default settings"), fintech ("ignore previous transaction instructions"). Use these instead of clinical notes as the running example.
- The domain-specific allowlisting section (FHIR URLs, clinical vocabulary) should become a general "domain adaptation" section with 2-3 brief examples across different industries, not a deep dive into one.

---

## Per-Post Issues

### Part 1: Zero-Touch LLM Guarding (`zero-touch-llm-guarding`)
- The "Why Not Just Use Middleware?" section is good motivation but makes the series feel LangChain-defensive. Consider tightening it — the key point is "my agents use raw LangGraph, middleware requires create_agent()" and that's one sentence.
- The five defense layers section is a summary of Part 2. If the posts merge, this becomes the natural structure.
- The "What This Doesn't Catch" section is honest and builds trust. Keep it.

### Part 2: Prompt Injection Patterns (`llm-guard-patterns`)
- Strongest technical content in the series. The regex patterns with comparison tables are the kind of concrete, useful detail that gets bookmarked.
- The base64 decoding section is well-done — the three details (whitespace tolerance, length floor, prefixed names) are the right level of depth.
- Random XML delimiters section is clear and practical.
- The `log_finding()` three-destination recording section is good architecture but could be shorter.

### Part 3: ML-Based Classifiers (`llm-guard-classifiers`)
- The classifier landscape comparison is useful. The three-row table (DeBERTa models, Prompt Guard 2, Model Armor) gives readers a quick orientation.
- Online vs offline deployment strategy is the key insight — 22M in the hot path, 86M on audit logs. This is practical and well-reasoned.
- The red teaming section (PromptFoo vs Garak) feels tacked on. It's useful information but doesn't flow naturally from the classifier discussion. Consider whether it belongs in a separate post or as part of a consolidated Part 1+2 under "testing your defenses."
- "The Full Stack" table at the end is a good summary of the entire series. If the series is restructured, this table should appear in the final post as a capstone.

### Part 4: LangChain Middleware (`llm-guard-middleware`)
- Too thin for a standalone post. The core content is: middleware can block (callbacks can't), here's the translation, use middleware if you use `create_agent()`.
- The comparison table (callbacks vs middleware) is useful but could be a section in Part 1.
- **Recommendation:** Fold into Part 1 or cut. If folded, the blocking mode via `wrap_model_call` is the only section that adds new information — the rest is mechanical translation.

### Healthcare Post (`llm-security-healthcare`)
- Best standalone writing in the series. Problem 3 (safety filters are backwards) and Problem 5 (failure mode is clinical) are the strongest sections.
- Problem 6 (multi-turn accumulation) is somewhat generic — it applies to any multi-turn agent, not just healthcare. Consider whether it belongs in a general post instead.
- The "Meta-Problem" conclusion section is effective but could be shorter.
- Per the author's positioning goals, this post should be reframed. See Strategic Direction section above.

### Template Method Post (`template-method-ml`)
- The thesis is sound but the post is imbalanced. The "Why Not the Others?" section (Strategy, Chain of Responsibility, Decorator, Builder) is ~60% of the content. Most readers with OOP experience already understand these patterns.
- The genuinely valuable content — the Go pipeline `Execute()` example, the Python `AuditedAgent` → `GuardedAgent` layered composition — is buried after the comparison section.
- "Where Template Method Doesn't Fit" section is good — it shows balanced thinking.
- **Recommendation:** Lead with the concrete examples (Go pipeline, Python agent layering). Collapse the pattern comparison into a summary table with one-line explanations. The post should be ~40% shorter.
- The Refactoring Guru link in references is attributed to Martin Fowler — Refactoring Guru is a separate site (by Alexander Shvets), not Fowler's work. Fix the attribution.

### Audit Logging Post (`zero-touch-llm-auditing`)
- Strong architectural content. The three-layer diagram and the invocation chain correlation section are the highlights.
- The "Alternatives I Considered" section (wrapping the client, manual calls, decorators, OTel) is well-structured and builds confidence in the design decision.
- The "Connecting Audit to Security" section at the end ties the audit and guard series together well. This should be preserved in any restructuring.
- The "Fail-Open" section is important and builds trust — the design choice to never let audit break the model is the right call, and explaining it shows production thinking.
- Two-tier recording (structured logs vs compliant store) repeats content from the healthcare post. Consolidate so one post is the canonical explanation.

---

## Writing and Voice Issues

### Inconsistent register
The healthcare post has personality ("It lasted about an hour"). The audit post is methodical and measured. The middleware post is dry. The template method post is occasionally defensive ("not as a buzzword, but as a practical engineering strategy", "that's obvious and unhelpful"). Aim for the healthcare post's register throughout — confident, direct, with occasional dry humor.

### Defensive phrasing
Several posts anticipate and preempt criticism rather than stating positions with confidence. Examples:
- "not as a buzzword, but as a practical engineering strategy"
- "that's obvious and unhelpful"
- "A small thing, but it prevents the inevitable typo"

**Recommendation:** Cut defensive qualifiers. State the position and move on. The reader will either agree or disagree — hedging doesn't help either way.

### The code is in an uncanny valley
The code is too specific to be pseudocode (real Python imports, real LangChain types, real class hierarchies) but too incomplete to run. There's no repo link, no installation instructions, no minimal working example. This creates ambiguity: is this extracted from a real codebase, or is it a design document?

**Recommendation:** Either link to a public repo (even a minimal reference implementation) or explicitly frame the code as "simplified from a production system" early in the series. The current state leaves the reader guessing.

---

## Suggested Series Structure (Post-Consolidation)

1. **"Zero-Touch LLM Security: Runtime Injection Defense Without Changing Your Model Code"** — Merged Parts 1+2. Architecture, patterns, structural isolation, output scanning. Include a "Middleware Alternative" section for create_agent() users. (~3000 words)
2. **"What Regex Can't Catch: ML-Based Injection Detection"** — Part 3, mostly as-is. Trim the red teaming section. (~2000 words)
3. **"The Template Method Pattern for ML Infrastructure"** — Restructured: lead with examples, compress pattern comparison. (~2000 words)
4. **"Zero-Touch Audit Logging for LLM Applications"** — Mostly as-is. Remove overlap with other posts. (~2500 words)
5. **"LLM Security in Regulated Domains"** — Reframed healthcare post. Broaden to "regulated domains" (healthcare, fintech, legal). Lead with the general inversions, use healthcare as the primary example but include brief fintech/legal parallels. (~2500 words)

This preserves all the technical content while reducing the post count from 7 to 5, eliminating repetition, and aligning with the author's positioning goals.

---

## Priority Actions (Ordered by Impact)

1. **Add production outcomes** to each post — rough numbers, surprises, tuning decisions
2. **Merge Parts 1+2** and fold Part 4 into the merged post
3. **Reframe healthcare examples** throughout the series to be domain-generic
4. **Restructure template method post** — lead with examples, compress comparisons
5. **Eliminate cross-post repetition** — one canonical explanation per concept
6. **Fix Refactoring Guru attribution** in template method references
7. **Normalize writing voice** across all posts
