# Explain it simply (your prep notes, not for evaluators)

This one's for you, not the submission - read it a couple of times before
you record. It's written so you could basically say it out loud. For the
shot-by-shot "click here, show this" list, use `docs/DEMO_SCRIPT.md`
instead - this one's about understanding it well enough to answer anything
that comes up, on camera or live in the interview.

## The one-paragraph version

*"This is an AI copilot for a plant operator. An alarm goes off on a
machine - say, a pump showing high vibration - and instead of the operator
digging through three different systems by hand, they type one sentence
and the AI investigates: it pulls the alarm data, scores how urgent it is,
checks the troubleshooting guides, checks whether a ticket for this already
exists, and writes up a draft incident report with everything cited. The
operator reads it, edits it if they want, and only then - with one
explicit click - does anything actually get written anywhere. The AI never
writes on its own."*

That's it. Everything else is detail in service of that one sentence.

## Walk through it with the real example (K-201)

Use this as your spine - it's concrete, and concrete beats abstract every
time on camera.

1. **You type:** *"Prepare an incident for the highest-priority active
   alarm in EastRefinery."*
2. **The AI doesn't know what that alarm is yet**, so its first move is to
   ask the Alarm API: "list me the active alarms in EastRefinery." That
   comes back with K-201, a high-vibration alarm on a pump.
3. **It asks a second question**: "how urgent is this one, on a 0-100
   scale?" - and a third: "what would an experienced operator usually do
   about this?" Each of these is a separate, specific request to the
   system - not the AI guessing from memory.
4. **It checks the knowledge base** - the plant's actual troubleshooting
   documents - for anything relevant to high vibration on this kind of
   asset, and gets back real passages with real document IDs attached.
5. **It checks whether a ticket already exists** for this, so you don't
   end up with two tickets for the same problem.
6. **It writes a draft** - what's wrong, why, what to do about it, how
   urgent - and every fact in that draft traces back to one of the steps
   above. Nothing in the draft is "the AI just knows this."
7. **You see the draft, plus a panel showing exactly which questions it
   asked and in what order** - that panel is the receipt. It's proof the
   AI actually looked things up rather than making a plausible-sounding
   paragraph.
8. **Nothing is written yet.** There's a checkbox. You have to tick it and
   confirm before a ticket gets created. If you never tick it, nothing
   happens - the draft just sits there for you to read.

## The big ideas, each with the analogy and the real reason

**MCP (the thing it uses to ask questions) - think of it like giving the
AI a fixed set of buttons to press, not free rein over the keyboard.** It
can press "look up this alarm" or "check priority," but it can't do
anything you didn't explicitly wire up a button for, and every button
press gets logged. That's what the trace panel is showing you - every
button it pressed, in order. The reason this matters: without it, you'd
have to just *trust* that the AI's answer about plant data is accurate,
with no way to check. With it, you can literally see the receipt.

**RAG / citations - like asking it to always show its homework.** When it
says "the troubleshooting guide recommends X," that's not it remembering
something from training - it actually searched the real documents at that
moment and is quoting a specific passage, which you can go look at
yourself. Why this matters specifically for a plant: a wrong answer about
dangerous equipment is not a minor inconvenience, so "trust me" isn't good
enough - "here's the exact paragraph I'm basing this on" is.

**The approval checkbox - a one-way door with a lock on it.** Looking
things up is reversible and harmless. Writing a new ticket into a real
system is not something you want an AI doing on its own judgment, even a
good one. So I split those two things apart on purpose: the AI can look
up as much as it wants freely, but the *one* action that changes something
real requires a human to explicitly say yes. And that check isn't just a
UI thing I could forget to show - it's built into the tool itself, so
there's no path through the system that skips it.

**Why it sometimes refuses to answer or has to retry - the honest one.**
I'm running this on a small model on my own laptop, not a giant cloud
model, so sometimes it genuinely struggles - it'll ask the same question
over and over, or take too long to respond. I built the system to notice
both of those and stop cleanly instead of getting stuck or making things
up to cover the gap. If that happens on camera, that's fine to show - it's
the system working as designed, not breaking.

## Anticipated questions (and the honest short answer)

**"Why go through MCP instead of just calling the API directly - isn't
that extra complexity?"**
> "Because I wanted a hard guarantee, not a convention I have to remember
> to follow. If the AI calls the API directly, there's nothing stopping a
> future version of this from skipping a step or calling something it
> shouldn't. Through MCP, the set of things it's *able* to do is fixed and
> declared up front, and every call is logged automatically - the
> guarantee is structural, not just 'I was careful.'"

**"How do you stop the AI from writing bad or wrong data into the
ticketing system?"**
> "Two layers. First, the only thing it's ever allowed to write is a
> ticket, and that requires an explicit human approval flag that only the
> UI sets, only after someone ticks a checkbox - the write tool itself
> refuses to run without it, so there's no way around it. Second, before
> it even gets to that point, I added a check that looks at the final
> draft and confirms every fact it's claiming actually came from a real
> lookup in this run, not something it just asserted."

**"What happens if the model hallucinates or makes something up?"**
> "I actually saw this happen during development - it cited two knowledge
> base articles it had never looked up. That's exactly why the grounding
> check exists: after it writes the draft, I compare every citation
> against what it actually looked up that run, and flag anything that
> doesn't match. It's not a guess about hallucination risk, it's a
> response to something I watched happen."

**"Why a local model instead of a cloud API like GPT-4?"**
> "Partly the assignment pointed that direction, but it's also a more
> honest test: handling a small model's real failure modes - getting
> stuck in a loop, timing out, running out of context - is a harder and
> more realistic engineering problem than building against a model that
> rarely misbehaves. It's also closer to how this would actually work in
> a real plant, where you often can't send operational data to an
> external cloud API at all."

**"What would you improve with more time?"**
> "Two honest ones: the local model's response time is inconsistent on
> CPU, which is a hardware constraint more than a design flaw, but it's
> still the rough edge a user would notice first. And right now this is
> single-user with no persistence across sessions - multi-user auth and a
> real audit log that survives a restart would be the next things I'd
> build."

## If you only remember three sentences

1. It can only touch real systems through a fixed, logged set of tools -
   it can't do anything undeclared, and you can always see exactly what it
   did.
2. Every fact in its draft is checked against something it actually looked
   up this run, not trusted on the model's word.
3. It can look things up freely, but it can never write anything without
   an explicit human click.
