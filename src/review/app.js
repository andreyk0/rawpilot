"use strict";
// Keys and clicks become commands for the server; the server answers with the page's #app
// element, which replaces the current one. The only state kept here is whether the help
// overlay is open (a class on <body>, outside #app, so a swap leaves it alone) and the queue
// of commands in flight.
(() => {
  const POLL_MS = 500;
  let queue = Promise.resolve();
  let pollTimer = null;
  // Commands queued or being sent, and a count that moves whenever one is queued or answered.
  // A poll's answer is dropped if either shows a command overlapped it: it may describe the
  // page from before that command, and the next command would then name the wrong shot.
  let inFlight = 0;
  let epoch = 0;

  const app = () => document.getElementById("app");

  function parse(html) {
    const holder = document.createElement("template");
    holder.innerHTML = html;
    return holder.content.firstElementChild;
  }

  function centreCurrent(root) {
    const current = root.querySelector(".thumb.current");
    if (current) current.scrollIntoView({ inline: "center", block: "nearest" });
  }

  // The page with the activity note emptied: what is left changes only when something the
  // photographer looks at changes.
  function withoutActivity(root) {
    const copy = root.cloneNode(true);
    copy.querySelector("#activity")?.replaceChildren();
    return copy.outerHTML;
  }

  function adopt(html, source) {
    const next = parse(html);
    if (!next) return;
    const old = app();
    if (source === "poll" && withoutActivity(old) === withoutActivity(next)) {
      // Only the progress moved: replacing the page would reload its pictures.
      const note = old.querySelector("#activity");
      const fresh = next.querySelector("#activity");
      if (note && fresh) note.replaceWith(fresh);
    } else {
      const strip = old.querySelector(".strip");
      const keep =
        source === "poll" && strip && old.dataset.shot === next.dataset.shot
          ? strip.scrollLeft
          : null;
      old.replaceWith(next);
      const fresh = next.querySelector(".strip");
      if (keep !== null && fresh) fresh.scrollLeft = keep;
      else centreCurrent(next);
    }
    schedulePoll();
  }

  function complain(text) {
    const line = document.createElement("div");
    line.className = "notice error";
    line.textContent = text;
    app().append(line);
  }

  // A key's command may hold {shot} and {viewed}: the shot on screen and the option shown
  // for it. They are filled from the page as it is when the command is sent, so a command
  // queued behind another acts on the page that one produced, and a page that is out of
  // date names the shot it shows, which the server then refuses. Null when the page has no
  // shot to name.
  function fill(template) {
    const { shot, viewed } = app().dataset;
    if (template.includes("{shot}") && shot === undefined) return null;
    if (template.includes("{viewed}") && viewed === undefined) return null;
    return template.replaceAll("{shot}", shot).replaceAll("{viewed}", viewed);
  }

  async function send(template) {
    try {
      const cmd = fill(template);
      if (cmd === null) return;
      const response = await fetch("/act", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: "cmd=" + encodeURIComponent(cmd),
      });
      if (response.ok) adopt(await response.text(), "command");
      else complain(`${response.status}: ${(await response.text()).trim()}`);
    } catch (error) {
      complain(`The server did not answer: ${error}`);
    } finally {
      inFlight -= 1;
      epoch += 1;
    }
  }

  function command(template) {
    inFlight += 1;
    epoch += 1;
    queue = queue.then(() => send(template));
  }

  async function refresh() {
    const started = epoch;
    try {
      const response = await fetch("/app");
      if (response.ok) {
        const html = await response.text();
        if (inFlight === 0 && epoch === started) adopt(html, "poll");
      }
    } catch (error) {
      // The next poll tries again.
    }
    schedulePoll();
  }

  // Polls while renders are owed: the page waits for the first, and the working view gains
  // shots and a progress count.
  function schedulePoll() {
    clearTimeout(pollTimer);
    if (app().dataset.polling !== undefined) {
      pollTimer = setTimeout(refresh, POLL_MS);
    }
  }

  function toggleHelp(open) {
    document.body.classList.toggle("help", open);
  }

  document.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.metaKey || event.altKey || event.repeat) return;
    if (document.body.classList.contains("help")) {
      toggleHelp(false);
      event.preventDefault();
      return;
    }
    const keys = JSON.parse(app().dataset.keys);
    if (!Object.hasOwn(keys, event.key)) return;
    event.preventDefault();
    const action = keys[event.key];
    if (action === "help") toggleHelp(true);
    else command(action);
  });

  document.addEventListener("click", (event) => {
    const target = event.target.closest("[data-cmd]");
    if (!target) return;
    event.preventDefault();
    command(target.dataset.cmd);
  });

  schedulePoll();
  centreCurrent(document);
})();
