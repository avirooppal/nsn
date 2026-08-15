const menuToggle = document.querySelector(".menu-toggle");
const mobileMenu = document.querySelector(".mobile-menu");
const mobileOverlay = document.querySelector(".mobile-overlay");
const mobileLinks = Array.from(document.querySelectorAll(".mobile-menu a"));
const navLinks = Array.from(document.querySelectorAll("[data-section]"));
const contentTargets = {
  trust: document.querySelector('[data-content="trust"]'),
  headlineOne: document.querySelector('[data-content="headline-one"]'),
  headlineTwo: document.querySelector('[data-content="headline-two"]'),
  subhead: document.querySelector('[data-content="subhead"]'),
  insightOne: document.querySelector('[data-content="insight-one"]'),
  insightTwo: document.querySelector('[data-content="insight-two"]'),
  insightThree: document.querySelector('[data-content="insight-three"]'),
  cta: document.querySelector('[data-content="cta"]'),
  command: document.querySelector('[data-content="command"]'),
};
const statsContainer = document.querySelector(".stats");
const commandButton = document.querySelector("[data-command-button]");
const githubUrl = "https://github.com/avirooppal/nsn";

const prefersReducedMotion = window.matchMedia(
  "(prefers-reduced-motion: reduce)"
).matches;

const sections = {
  home: {
    trust: "Trusted with SLMs and AI Agents",
    headlineOne: "Intelligence",
    headlineTwo: "Designed To Evolve",
    subhead:
      "NeuroSleepNet gives any language model or AI agent persistent, long-term memory with two lines of code, running entirely locally.",
    insights: ["Local memory", "Hybrid retrieval", "Sleep consolidation"],
    cta: "Get Started",
    command: "git clone https://github.com/avirooppal/nsn.git",
    stats: [
      ["<", 2, " lines", 0, "Integration"],
      ["%", 100, "%", 0, "Conflict Recall"],
      ["*", 85.71, "%", 2, "Update Recall"],
      ["#", 42.67, "%", 2, "Multi-Hop Recall"],
    ],
  },
  product: {
    trust: "Memory OS for Local Agents",
    headlineOne: "Persistent Memory",
    headlineTwo: "In Two Lines",
    subhead:
      "Wrap any callable model, OpenAI-compatible client, LangChain chain, AutoGen worker, CrewAI agent, or FastAPI service and NSN recalls, injects, stores, classifies, and consolidates memory automatically.",
    insights: ["Episodic memory", "Semantic memory", "Procedural memory"],
    cta: "Explore Product",
    command: "pip install -e .",
    stats: [
      ["<", 5, " memories", 0, "Injected Context"],
      ["%", 0, " cloud", 0, "External APIs"],
      ["*", 3, " types", 0, "Memory Classes"],
      ["#", 0.95, "", 2, "Dedupe Threshold"],
    ],
  },
  results: {
    trust: "Research-Grade Benchmark Results",
    headlineOne: "Measured Recall",
    headlineTwo: "Across Hard Tasks",
    subhead:
      "In seed-42 head-to-head evaluation, NSN outperformed six memory baselines on temporal knowledge updates, source-trust contradiction resolution, and multi-hop relational traversal.",
    insights: ["+10.71 pp update recall", "+75.00 pp conflict recall", "2x multi-hop recall"],
    cta: "View Results",
    command: "python -m benchmarks.run_head_to_head --samples 20 --chains 10 --seed 42",
    stats: [
      ["<", 85.71, "%", 2, "Knowledge Update"],
      ["%", 100, "%", 0, "Contradictions"],
      ["*", 1, " MRR", 3, "Gold Rank Quality"],
      ["#", 42.67, "%", 2, "Multi-Hop Recall"],
    ],
  },
  architecture: {
    trust: "FAISS + FTS5 + Graph + Sleep",
    headlineOne: "Memory That",
    headlineTwo: "Consolidates Offline",
    subhead:
      "NSN combines dense semantic search, SQLite FTS keyword search, entity graph traversal, RRF fusion, trust scoring, NREM synthesis, REM contradiction resolution, and importance decay.",
    insights: ["Hybrid search", "TrustManager", "NREM + REM sleep"],
    cta: "See Architecture",
    command: "uvicorn neurosleepnet.integrations.api:app --host 0.0.0.0 --port 8000",
    stats: [
      ["<", 0.85, "x", 2, "Sleep Decay"],
      ["%", 1, "", 1, "System Trust"],
      ["*", 3, " paths", 0, "Search Fusion"],
      ["#", 7, " systems", 0, "Benchmarked"],
    ],
  },
};

function easeOutCubic(t) {
  return 1 - Math.pow(1 - t, 3);
}

function formatStat(value, decimals, suffix) {
  return `${value.toFixed(decimals)}${suffix}`;
}

function countStat(stat, index) {
  if (stat.dataset.counted === "true") {
    return;
  }

  stat.dataset.counted = "true";

  const valueEl = stat.querySelector(".stat-value");
  const target = Number(stat.dataset.target);
  const suffix = stat.dataset.suffix || "";
  const decimals = Number(stat.dataset.decimals || 0);

  if (prefersReducedMotion) {
    valueEl.textContent = formatStat(target, decimals, suffix);
    return;
  }

  const duration = 1500 + index * 80;
  const delay = 120 + index * 70;
  const startTime = performance.now() + delay;

  function tick(now) {
    const progress = Math.min(Math.max((now - startTime) / duration, 0), 1);
    const current = target * easeOutCubic(progress);
    valueEl.textContent = formatStat(current, decimals, suffix);

    if (progress < 1) {
      requestAnimationFrame(tick);
    } else {
      valueEl.textContent = formatStat(target, decimals, suffix);
    }
  }

  requestAnimationFrame(tick);
}

function renderStats(items) {
  statsContainer.innerHTML = items
    .map(
      ([icon, target, suffix, decimals, label], index) => `
        <div
          class="stat content-swap"
          style="--d: ${0.5 + index * 0.08}s"
          data-target="${target}"
          data-suffix="${suffix}"
          data-decimals="${decimals}"
        >
          <div class="stat-icon">${icon === "<" ? "&lt;" : icon}</div>
          <div class="stat-value">${formatStat(0, decimals, suffix)}</div>
          <div class="stat-label">${label}</div>
        </div>
      `
    )
    .join("");

  Array.from(document.querySelectorAll(".stat")).forEach(countStat);
}

function animateContent(elements) {
  if (prefersReducedMotion) {
    return;
  }

  elements.forEach((element) => {
    element.classList.remove("content-swap");
    void element.offsetWidth;
    element.classList.add("content-swap");
  });
}

function setActiveSection(sectionName) {
  const section = sections[sectionName] || sections.home;

  contentTargets.trust.textContent = section.trust;
  contentTargets.headlineOne.textContent = section.headlineOne;
  contentTargets.headlineTwo.textContent = section.headlineTwo;
  contentTargets.subhead.textContent = section.subhead;
  contentTargets.insightOne.textContent = section.insights[0];
  contentTargets.insightTwo.textContent = section.insights[1];
  contentTargets.insightThree.textContent = section.insights[2];
  contentTargets.cta.textContent = section.cta;
  contentTargets.cta.href = githubUrl;
  contentTargets.command.textContent = section.command;

  navLinks.forEach((link) => {
    link.classList.toggle("active", link.dataset.section === sectionName);
  });

  renderStats(section.stats);
  animateContent(Object.values(contentTargets));
}

async function copyCommand() {
  const command = contentTargets.command.textContent.trim();

  try {
    await navigator.clipboard.writeText(command);
    commandButton.classList.add("copied");
    commandButton.setAttribute("aria-label", "Command copied");
    setTimeout(() => {
      commandButton.classList.remove("copied");
      commandButton.setAttribute("aria-label", "Copy command");
    }, 1300);
  } catch (error) {
    window.prompt("Copy this command:", command);
  }
}

function openMenu() {
  document.body.classList.add("menu-open");
  menuToggle.setAttribute("aria-expanded", "true");
  menuToggle.setAttribute("aria-label", "Close menu");
  mobileMenu.hidden = false;
  mobileOverlay.hidden = false;
}

function closeMenu() {
  document.body.classList.remove("menu-open");
  menuToggle.setAttribute("aria-expanded", "false");
  menuToggle.setAttribute("aria-label", "Open menu");
  mobileMenu.hidden = true;
  mobileOverlay.hidden = true;
}

navLinks.forEach((link) => {
  link.addEventListener("click", (event) => {
    event.preventDefault();
    const section = link.dataset.section;
    history.replaceState(null, "", `#${section}`);
    setActiveSection(section);
    closeMenu();
  });
});

menuToggle.addEventListener("click", () => {
  if (document.body.classList.contains("menu-open")) {
    closeMenu();
  } else {
    openMenu();
  }
});

mobileOverlay.addEventListener("click", closeMenu);
mobileLinks.forEach((link) => link.addEventListener("click", closeMenu));
commandButton.addEventListener("click", copyCommand);

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeMenu();
  }
});

window.addEventListener("resize", () => {
  if (window.innerWidth > 720) {
    closeMenu();
  }
});

setActiveSection(location.hash.replace("#", "") || "home");
