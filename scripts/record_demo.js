// Drives the real yes, if app in Chrome on this laptop and records it.
// Usage: node record_demo.js [baseUrl] [outDir] [narration.json]
// narration.json maps caption text -> {sec}; captions then stay up until the line is spoken,
// and the times each caption appeared are written to <outDir>/captions.json for mixing audio.
const { chromium } = require("playwright");

const BASE = process.argv[2] || "http://127.0.0.1:8000";
const OUT = process.argv[3] || __dirname + "/video";
const NARR = process.argv[4] ? require(require("path").resolve(process.argv[4])) : {};
const W = 1280, H = 720;
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

// Caption bar + visible cursor, injected on every page load (Playwright video has no cursor).
const OVERLAY = `(() => {
  const add = () => {
    if (document.getElementById("__cap")) return;
    const c = document.createElement("div"); c.id = "__cap";
    c.style.cssText = "position:fixed;left:50%;bottom:28px;transform:translateX(-50%);z-index:2147483647;max-width:80%;padding:12px 22px;border-radius:12px;background:rgba(10,12,16,.88);color:#fff;font:600 19px/1.35 system-ui,sans-serif;box-shadow:0 8px 30px rgba(0,0,0,.35);opacity:0;transition:opacity .3s;pointer-events:none;text-align:center";
    document.body.appendChild(c);
    const m = document.createElement("div"); m.id = "__cur";
    m.style.cssText = "position:fixed;left:0;top:0;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;background:rgba(255,196,0,.55);border:2px solid #ffb300;z-index:2147483647;pointer-events:none;transition:transform .08s";
    document.body.appendChild(m);
    addEventListener("mousemove", (e) => { m.style.left = e.clientX + "px"; m.style.top = e.clientY + "px"; }, true);
    addEventListener("mousedown", () => (m.style.transform = "scale(.7)"), true);
    addEventListener("mouseup", () => (m.style.transform = ""), true);
  };
  window.__caption = (t) => { add(); const c = document.getElementById("__cap"); c.textContent = t; c.style.opacity = t ? "1" : "0"; };
  if (document.body) add(); else addEventListener("DOMContentLoaded", add);
})();`;

(async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: false, args: [`--window-size=${W},${H + 90}`] });
  const ctx = await browser.newContext({ viewport: { width: W, height: H }, recordVideo: { dir: OUT, size: { width: W, height: H } } });
  await ctx.addInitScript(OVERLAY);
  const page = await ctx.newPage();
  const t0 = Date.now(), shown = [];
  page.on("console", (m) => m.type() === "error" && console.log("console:", m.text()));
  page.on("pageerror", (e) => console.log("pageerror:", e.message));
  page.on("framenavigated", (f) => f === page.mainFrame() && console.log("nav:", f.url()));
  const say = async (t, ms = 3200) => {
    await page.evaluate((x) => window.__caption(x), t);
    if (t) shown.push({ text: t, at: (Date.now() - t0) / 1000 });
    await wait(Math.max(ms, NARR[t] ? NARR[t].sec * 1000 + 500 : 0));
  };
  const click = async (sel, ms = 1200) => {
    const el = page.locator(sel).first();
    await el.scrollIntoViewIfNeeded();
    const b = await el.boundingBox();
    if (b) await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2, { steps: 18 });
    await el.click();
    await wait(ms);
  };
  const type = async (sel, v) => { await click(sel, 200); await page.fill(sel, ""); await page.locator(sel).pressSequentially(String(v), { delay: 90 }); await wait(400); };
  const slide = async (sel, v) => {
    await page.locator(sel).scrollIntoViewIfNeeded();
    const b = await page.locator(sel).boundingBox();
    const min = +(await page.getAttribute(sel, "min")), max = +(await page.getAttribute(sel, "max"));
    const cur = +(await page.inputValue(sel));
    const x = (n) => b.x + 8 + (b.width - 16) * (n - min) / (max - min), y = b.y + b.height / 2;
    await page.mouse.move(x(cur), y, { steps: 12 }); await page.mouse.down();
    await page.mouse.move(x(v), y, { steps: 30 }); await page.mouse.up();
    await wait(1600);
  };
  const scroll = async (dy, ms = 1500) => { await page.mouse.wheel(0, dy); await wait(ms); };

  // 1. Product site
  await page.goto(BASE + "/", { waitUntil: "networkidle" });
  await say("yes, if: a counterfactual payment risk engine, running live on this laptop", 3800);
  await scroll(700); await scroll(700);
  await say("Risk engines stop at a score. yes, if finds the cheapest change that turns a risky payment into a yes.", 4000);

  // 2. Live decision
  await page.goto(BASE + "/app#decision", { waitUntil: "networkidle" });
  await say("Live decision: each payment is scored by a calibrated LightGBM model", 3000);
  await click('#sample-list [data-i="0"]', 2500);
  await say("A regular purchase on a known device is approved straight away", 2600);
  await click('#sample-list [data-i="3"]', 2500);
  await page.locator("#result").scrollIntoViewIfNeeded(); await wait(1500);
  await say("A stolen card on a new device at night: high risk, with the reasons in plain language", 3600);
  await click('#sample-list [data-i="2"]', 2500);
  await page.locator("#result").scrollIntoViewIfNeeded(); await wait(1200);
  await say("Just over the line: the engine offers 'yes, if' changes that would get it approved", 3800);
  await page.locator("#custom").scrollIntoViewIfNeeded();
  await say("Or score a custom payment, typed in by hand", 2200);
  await type("#f-amt", 1450);
  await type("#f-shared", 7);
  await page.selectOption("#f-email", "anonmail.net"); await wait(600);
  await click('#custom button[type="submit"]', 2500);
  await page.locator("#result").scrollIntoViewIfNeeded(); await wait(2500);

  // 3. Counterfactual explorer
  await page.goto(BASE + "/app#explorer", { waitUntil: "networkidle" }); await wait(1500);
  await say("Counterfactual explorer: change the payment and watch the risk move", 3000);
  await slide("#x-amt", 40);
  await say("Paying 40% now instead of the full amount", 2400);
  await slide("#x-delay", 12);
  await say("…and letting it go through 12 hours later", 2400);
  await click('#x-card6 [data-v="debit"]', 1800);
  const apply = page.locator("#x-options [data-apply]");
  if (await apply.count()) { await say("Apply the engine's cheapest suggestion in one click", 2000); await click("#x-options [data-apply]", 2500); }

  // 4. PayPal sandbox checkout
  await page.goto(BASE + "/app#checkout", { waitUntil: "networkidle" }); await wait(2500);
  await say("Sandbox checkout: the engine decides before PayPal ever opens", 3200);
  const personas = await page.locator("#personas button").count();
  for (let i = 0; i < personas; i++) {
    await click(`#personas button >> nth=${i}`, 2600);
    await page.locator("#buyer").scrollIntoViewIfNeeded(); await wait(1000);
  }
  await say("Buyer sees coarse guidance; the merchant sees the risk, reasons and the PayPal plan", 3800);
  const offers = page.locator("#buyer .offers button");
  if (await offers.count()) {
    await click("#buyer .offers button", 2000);
    await page.locator("#paypal-buttons").scrollIntoViewIfNeeded();
    await say("Picking a 'yes, if' unlocks the PayPal sandbox button for this order", 3500);
  }

  // 5. Research panels
  for (const [tab, text] of [
    ["policy", "Policy battle: the learned friction policy adapts when step-up stops working"],
    ["adversary", "Adversary arena: retraining cuts attacker evasion from 28% to under 9%"],
    ["network", "Network view: one device, many cards, the ring signal"],
    ["model", "Model card: metrics, calibration and fairness by segment"],
  ]) {
    await click(`a[href="#${tab}"]`, 1500);
    await say(text, 3000);
    await scroll(500, 1800);
  }

  // 6. API docs
  await page.goto(BASE + "/docs", { waitUntil: "networkidle" });
  await say("Everything is also available as an API for merchants", 3000);
  await scroll(600, 2000);
  await say("", 600);

  await ctx.close();
  require("fs").writeFileSync(OUT + "/captions.json", JSON.stringify(shown, null, 1));
  await browser.close();
  console.log("saved to", OUT);
})().catch(async (e) => { console.error(e.message.split("\n")[0]); process.exit(1); });
