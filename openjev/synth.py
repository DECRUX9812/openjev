"""openjev/synth/generate.py — synthetic decision-data generator, stage 1+2.

Two stages per batch, run as separate passes so either can be re-run:
  gen:   spec matrix -> raw postings {title, employer, pay} + intended bucket (teacher: gemini-3.6-flash-high)
  label: postings -> typed decisions from the SAME 7-question rubric the lab sends Jev (teacher: claude-sonnet-4-6)

Provenance, intended bucket and the teacher's rationale are kept in the record; Jev
verification and the agreement filter live in verify.py.
"""
from __future__ import annotations

import argparse, json, os, random, re, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"; DATA.mkdir(parents=True, exist_ok=True)
LAB = Path(os.environ.get("TYPESAFE_LAB", "/home/decrux/Code/typesafe-lab"))
PROXY = os.environ.get("CLIPROXY_URL", "http://127.0.0.1:8317/v1/chat/completions")
KEY = None

def api_key():
    global KEY
    if KEY: return KEY
    for line in (Path.home() / ".hermes" / ".env").read_text().splitlines():
        if line.startswith("CLIPROXYAPI_API_KEY="):
            KEY = line.split("=", 1)[1].strip()
    return KEY

def call(model, system, user, *, max_tokens=2000, temperature=1.0, retries=3):
    payload = {"model": model, "messages": [{"role": "system", "content": system},
                                            {"role": "user", "content": user}],
               "max_tokens": max_tokens, "temperature": temperature}
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(PROXY, data=json.dumps(payload).encode(),
                                         headers={"Authorization": f"Bearer {api_key()}",
                                                  "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                body = json.load(r)
            return body["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            last = exc; time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"call failed: {last}")

def extract_json(text):
    """Pull the first JSON object/array out of a model reply."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence: text = fence.group(1).strip()
    for open_c, close_c in (("[", "]"), ("{", "}")):
        i = text.find(open_c)
        if i == -1: continue
        depth = 0
        for j in range(i, len(text)):
            if text[j] == open_c: depth += 1
            elif text[j] == close_c:
                depth -= 1
                if depth == 0:
                    return json.loads(text[i:j + 1])
    raise ValueError("no JSON found")

# ---------------------------------------------------------------- spec matrix
FAMILIES = {
    "it_infra": ["IT Support Analyst", "Network Administrator", "Systems Administrator", "Help Desk Technician",
                 "Infrastructure Analyst", "Server Administrator", "IT Coordinator"],
    "software": ["Software Developer", "Web Developer", "Full Stack Developer", "Application Analyst",
                 "Programmer Analyst", "Mobile Developer", "DevOps Engineer"],
    "data": ["Data Analyst", "Business Intelligence Analyst", "Data Engineer", "Reporting Analyst",
             "Information Management Analyst", "Data Architect", "Analytics Coordinator"],
    "web_digital": ["Web Designer", "Digital Marketing Coordinator", "Social Media Manager",
                    "E-Commerce Coordinator", "Graphic Designer", "Website Administrator"],
    "security": ["Cybersecurity Analyst", "Security Administrator", "Information Security Officer", "IT Auditor"],
    "grc_analyst": ["Policy Analyst", "Business Analyst", "Project Coordinator", "Change Management Analyst",
                    "Records Management Clerk", "Privacy Analyst"],
    "trades": ["Welder", "Carpenter", "Concrete Finisher", "Heavy Duty Mechanic", "Electrician",
               "Plumber", "Roofer", "Ironworker", "Millwright"],
    "care_health": ["Continuing Care Assistant", "Registered Nurse", "Health Care Aide", "Licensed Practical Nurse",
                    "Physician Assistant", "Medical Office Assistant", "Pharmacy Assistant"],
    "retail_food": ["Cashier", "Line Cook", "Assistant Cook", "Server", "Baker", "Retail Sales Associate",
                    "Store Manager", "Barista", "Dishwasher"],
    "driving_logistics": ["Class 1 Driver", "Delivery Driver", "Warehouse Associate", "Forklift Operator",
                          "Dispatcher", "Yard Attendant", "Long Haul Truck Driver"],
    "education": ["Elementary Teacher", "Educational Assistant", "Early Childhood Educator", "Instructor",
                  "Tutor", "School Bus Driver"],
    "office_admin": ["Administrative Assistant", "Office Administrator", "Receptionist", "Executive Assistant",
                     "Data Entry Clerk", "Accounting Clerk", "Payroll Administrator"],
    "finance_legal": ["Accountant", "Bookkeeper", "Legal Assistant", "Paralegal", "Financial Analyst",
                      "Articling Student", "Tax Preparer"],
    "sales_mgmt": ["Sales Representative", "Territory Manager", "Operations Manager", "Branch Manager",
                   "Business Development Manager", "General Manager", "Director of Operations"],
    "public_sector": ["Program Coordinator", "Community Relations Coordinator", "By-Law Officer",
                      "Communications Officer", "Recreation Programmer", "Case Manager"],
    "multi_hat_smallbiz": ["Office Administrator & Social Media", "Marketing and Website Coordinator",
                           "Bookkeeper with IT Duties", "Administrative & IT Support", "Operations & Systems Coordinator",
                           "Marketing Coordinator (Website + Socials)"],
}
EMPLOYER_TYPES = {
    "small_business": ["{name} Ltd.", "{name} Inc.", "{name} & Sons", "{name} Contracting", "{name} Services"],
    "larger_firm": ["Prairie {name} Group", "{name} Corporation", "{name} Solutions", "{name} Health Region"],
    "public": ["City of {city}", "{name} School Division", "Saskatchewan {name} Authority", "Sask {name}",
               "{name} Public Library"],
    "nonprofit": ["{name} Community Association", "{name} Food Bank", "{name} Housing Society", "{name} Outreach"],
    "staffing": ["{name} Staffing Solutions", "{name} Recruitment Group", "Express {name} Agency"],
    "mlm_commission": ["{name} Independent Distributors", "Team {name} Marketing", "{name} Sales Partners"],
}
NAME_STEMS = ["Northgate", "Prairie Sky", "Cedar Ridge", "Wheatland", "Qu'Appelle", "Buffalo Pound", "Wascana",
              "Moose Jaw", "Regina Beach", "Pilot Butte", "Deer Valley", "Sunrise", "Legacy", "Harvest",
              "Maple Creek", "Stonegate", "Riverside", "Lakeshore", "Aspen", "Pinewood"]
CITIES = ["Regina", "Saskatoon", "Moose Jaw", "Prince Albert", "Yorkton", "Swift Current", "Weyburn", "Estevan"]
PAY_STYLES = ["none", "hourly", "salary", "range", "vague"]
TACTICS = ["plain", "numbered_posting_id", "evergreen", "urgent", "graveyard_shift", "part_time",
           "temp_contract", "vague_one_liner", "bilingual", "unionized"]

def spec_batch(n, rng):
    specs = []
    weights = {"it_infra": 10, "software": 8, "data": 8, "web_digital": 7, "security": 4, "grc_analyst": 6,
               "trades": 10, "care_health": 8, "retail_food": 9, "driving_logistics": 7, "education": 6,
               "office_admin": 8, "finance_legal": 7, "sales_mgmt": 7, "public_sector": 5, "multi_hat_smallbiz": 5}
    fams = list(weights); w = [weights[f] for f in fams]
    for _ in range(n):
        fam = rng.choices(fams, weights=w)[0]
        title = rng.choice(FAMILIES[fam])
        etype = rng.choices(list(EMPLOYER_TYPES), weights=[6, 5, 4, 3, 3, 2][:len(EMPLOYER_TYPES)])[0]
        spec = {
            "family": fam, "title": title, "employer_type": etype,
            "employer": rng.choice(EMPLOYER_TYPES[etype]).format(name=rng.choice(NAME_STEMS),
                                                                 city=rng.choice(CITIES)),
            "city": rng.choice(CITIES), "pay_style": rng.choice(PAY_STYLES),
            "tactic": rng.choices(TACTICS, weights=[6, 2, 2, 2, 1, 2, 2, 2, 1, 1])[0],
        }
        specs.append(spec)
    return specs

GEN_SYSTEM = """You write realistic Saskatchewan job postings for a machine-learning dataset.
Style notes from real SaskJobs postings:
- Titles are short and blunt; some carry a numeric posting id ("100970 - Information Management Analyst").
- Employer names are local businesses, school divisions, health regions, municipalities or staffing agencies.
- "pay" is often EMPTY; when present it looks like "$18.50 hourly", "$52,000 - $64,000 annually", "Commensurate with experience".
- Evergreen/staffing ads say "always hiring", "multiple positions", "great opportunity to join our talent pool".
- Commission-only / MLM ads push "unlimited earning potential", "be your own boss", "no experience necessary".
Return ONLY a JSON array, no prose. One object per line item in the request, exact keys: title, employer, pay."""

def gen_batch(specs, rng, attempt=1):
    lines = []
    for i, s in enumerate(specs):
        lines.append(f"{i+1}. family={s['family']} | role hint={s['title']} | employer type={s['employer_type']} "
                     f"| employer={s['employer']} | city={s['city']} | pay style={s['pay_style']} | style={s['tactic']}")
    user = ("Write one posting per line item, keeping the requested family and employer, and choose realistic "
            "title/employer/pay wording. Vary wording across items.\n\n" + "\n".join(lines))
    raw = call("gemini-3.6-flash-high", GEN_SYSTEM, user, max_tokens=4000, temperature=1.0)
    items = extract_json(raw)
    out = []
    for s, it in zip(specs, items):
        if not isinstance(it, dict): continue
        out.append({**s, "title": str(it.get("title", s["title"]))[:120],
                    "employer": str(it.get("employer", s["employer"]))[:120],
                    "pay": str(it.get("pay") or "")[:60]})
    return out

# ---------------------------------------------------------------- labelling
LABEL_SYSTEM = """You classify Saskatchewan job postings for a one-person IT & AI services firm in Regina.
Buckets are DISJOINT — pick the single best one:
- service_lead: a business with a concrete, small-scope technical or digital need (website, IT support, automation, data/reporting, security) that a one-person firm could be hired to deliver — including a small business hiring for exactly that work.
- staff_role: a real organization filling a regular employee seat whose own work is technical or digital (IT, software, data, security, systems, network).
- generic_job: any other real posting — trades, care, retail, food, driving, health, education, or office/clerical/legal/finance/sales/management roles with no technical content.
- junk: not a real buyer of work — commission-only or MLM, unpaid/volunteer, a mass-reposted staffing ad, or too vague to be a real job at all.
  Evergreen wording ("talent pool", "always hiring", "ongoing recruitment") alone does NOT make a posting junk — judge the underlying role first; only use junk when the posting itself is not a real job offer.
Also answer: technical_need, business_buyer, small_firm_doable, pay_stated, evergreen_repost (booleans), and fit 0-4
(0 nothing, 1 weak, 2 watchlist, 3 pitch, 4 pitch now).
Return ONLY JSON: {"bucket": "...", "technical_need": bool, "business_buyer": bool, "small_firm_doable": bool,
"pay_stated": bool, "evergreen_repost": bool, "fit": int, "confidence": 0-1, "rationale": "one short line"}"""

def label_batch(items):
    user = "Postings:\n" + "\n".join(
        f"{i+1}. title: {it['title']}\n   employer: {it['employer']}\n   pay: {it.get('pay') or ''}"
        for i, it in enumerate(items))
    raw = call("claude-sonnet-4-6", LABEL_SYSTEM, user, max_tokens=4000, temperature=0.0)
    parsed = extract_json(raw)
    out = []
    for it, lab in zip(items, parsed):
        if not isinstance(lab, dict): continue
        out.append({**it, "label": lab})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "label"], required=True)
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--in-file", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    lock = threading.Lock()
    t0 = time.time()

    if args.stage == "gen":
        out_path = Path(args.out or DATA / "raw_postings.jsonl")
        done = 0
        def work(chunk):
            nonlocal done
            for attempt in range(3):
                try:
                    rows = gen_batch(chunk, rng, attempt)
                    with lock:
                        with out_path.open("a") as fh:
                            for r in rows: fh.write(json.dumps(r) + "\n")
                        done += len(rows)
                        print(f"[gen] +{len(rows)} (total {done}) {time.time()-t0:.0f}s", flush=True)
                    return
                except Exception as exc:  # noqa: BLE001
                    print(f"[gen] retry {attempt+1}: {str(exc)[:120]}", flush=True); time.sleep(3)
            print("[gen] chunk failed", flush=True)
        specs = spec_batch(args.count, rng)
        chunks = [specs[i:i + args.batch] for i in range(0, len(specs), args.batch)]
        with ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(work, chunks))
    else:
        src = Path(args.in_file or DATA / "raw_postings.jsonl")
        rows = [json.loads(l) for l in src.read_text().splitlines()]
        out_path = Path(args.out or DATA / "labelled.jsonl")
        # resumable: skip rows already labelled (matched on the posting itself)
        if out_path.exists():
            seen = set()
            for line in out_path.read_text().splitlines():
                try:
                    r = json.loads(line)
                    seen.add((r.get("title"), r.get("employer"), r.get("pay")))
                except Exception:  # noqa: BLE001
                    pass
            before = len(rows)
            rows = [r for r in rows if (r.get("title"), r.get("employer"), r.get("pay")) not in seen]
            print(f"[label] resuming: {before - len(rows)} already labelled, {len(rows)} to go", flush=True)
        chunks = [rows[i:i + args.batch] for i in range(0, len(rows), args.batch)]
        done = 0
        def work2(chunk):
            nonlocal done
            for attempt in range(3):
                try:
                    labelled = label_batch(chunk)
                    with lock:
                        with out_path.open("a") as fh:
                            for r in labelled: fh.write(json.dumps(r) + "\n")
                        done += len(labelled)
                        print(f"[label] +{len(labelled)} (total {done}/{len(rows)}) {time.time()-t0:.0f}s", flush=True)
                    return
                except Exception as exc:  # noqa: BLE001
                    print(f"[label] retry {attempt+1}: {str(exc)[:120]}", flush=True); time.sleep(3)
            print("[label] chunk failed", flush=True)
        with ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(work2, chunks))

    print(f"[done] stage={args.stage} in {time.time()-t0:.0f}s -> {out_path}", flush=True)

if __name__ == "__main__":
    main()
