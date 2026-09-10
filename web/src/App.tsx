import { ABLATION, CAVEAT, DATASET, HEADLINE, HIVES, PRIOR, STACK } from './data'

const TOC = [
  { href: '#problem', label: 'The problem' },
  { href: '#protocol', label: 'How I test it' },
  { href: '#result', label: 'The result' },
  { href: '#temp', label: 'The temperature sensor' },
  { href: '#scorecard', label: 'How well it works' },
  { href: '#decisions', label: 'What I skipped' },
  { href: '#stack', label: 'The tools' },
]

// A horizontal MAE bar. Lower is better; longer bar = worse error.
function MaeBar({ label, mae, ci, tone }: { label: string; mae: number; ci?: [number, number]; tone: string }) {
  const max = 14
  const pct = Math.min(100, (mae / max) * 100)
  return (
    <div className="bar-row">
      <div className="bar-row__label">{label}</div>
      <div className="bar-track">
        <div className={`bar-fill bar-fill--${tone}`} style={{ width: `${pct}%` }} />
      </div>
      <div className="bar-row__num">
        {mae.toFixed(2)}
        {ci ? <span className="ci"> ±[{ci[0]}, {ci[1]}]</span> : null}
      </div>
    </div>
  )
}

function Stat({ value, label, tone }: { value: string; label: string; tone: string }) {
  return (
    <div className={`stat stat--${tone}`}>
      <div className="stat__value">{value}</div>
      <div className="stat__label">{label}</div>
    </div>
  )
}

function Beat({ id, kicker, title, children }: { id: string; kicker: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="beat">
      <p className="beat__kicker">{kicker}</p>
      <h2 className="beat__title">{title}</h2>
      <div className="beat__body">{children}</div>
    </section>
  )
}

export function App() {
  return (
    <>
      <a className="skip-link" href="#problem">Skip to the walkthrough</a>

      <div className="masthead">
        <div className="masthead__inner">
          <div className="masthead__mark"><b>HiveHum</b> · a walkthrough</div>
          <ul className="masthead__nav">
            <li><a href="#problem">problem</a></li>
            <li><a href="#result">result</a></li>
            <li><a href="#temp">temperature</a></li>
            <li><a href="#stack">tools</a></li>
          </ul>
        </div>
      </div>

      <main className="page">
        <header className="page-hero">
          <p className="meta">A walkthrough of predicting colony strength from in-hive audio</p>
          <h1>HiveHum</h1>
          <p className="lead">
            A <strong>frame of bees</strong> is one of the wooden frames that hang inside a hive box.
            When a beekeeper opens a hive, they count how many frames are covered in bees, and that
            count is how they judge whether the colony is strong. I wanted to see whether I could
            predict that count from sound alone, using a microphone left running inside the hive.
          </p>
          <p className="lead">
            But how you test it decides whether the answer means anything. So I test on a hive the
            model has never listened to, because that is the only way to tell it learned what bees
            sound like instead of memorising one particular hive.
          </p>
          <p className="intro-detail">
            This is a portfolio project, not a bee-monitoring tool. It runs on one public dataset:{' '}
            {DATASET.hives} rooftop hives recorded in {DATASET.city} across {DATASET.years}. The
            numbers you see are real output from the code in this repo, but I ran it on a small slice
            of the data, so read them as a demonstration of the method rather than a score to trust.{' '}
            {CAVEAT}
          </p>
          <nav aria-label="On this page">
            <ul className="toc">
              {TOC.map((i) => (
                <li key={i.href}><a href={i.href}>{i.label}</a></li>
              ))}
            </ul>
          </nav>
        </header>

        <Beat id="problem" kicker="The problem" title="A model can look great by cheating">
          <p>
            Here is the trap I was worried about. The audio is dense: there is a recording every 30
            minutes. The labels are sparse: a beekeeper only inspects a hive about once every two
            weeks. If I throw all the clips into one pile and split them at random into training and
            testing, the model can score well without learning anything about bees. It just learns to
            recognise each hive. Every hive has its own background hum, its own spot on a noisy city
            rooftop, its own weather, and recognising the hive is not the same as hearing how strong
            the colony is.
          </p>
          <p className="aside">
            This is already known. A follow-up study on the same dataset ({PRIOR.cite}) reported the
            correlation between predicted and real colony strength falling from about{' '}
            <b>{PRIOR.randomR}</b> on a random split to about <b>{PRIOR.hiveIndependentR}</b> once the
            hives were kept apart. So I am not claiming to find the problem. I am reproducing it on a
            small, honest slice and showing you what it looks like.
          </p>
        </Beat>

        <Beat id="protocol" kicker="How I test it" title="Test on a hive it never heard">
          <p>
            <strong>Leave-one-hive-out</strong> means what it says. I take one hive out of the
            training data completely, train on the rest, then ask the model to predict the hive it
            never heard, a bit like asking a doctor to assess a patient they have never met. I compare
            that against the easier test, where the model has already heard the same hive on its
            earlier days. Same hive, two ways, both scored on the exact same held-out days ({HEADLINE.nTest}{' '}
            of them, for hive {HEADLINE.testId}). The distance between the two scores is how much the
            model was leaning on knowing the hive.
          </p>
          <div className="hives-strip">
            {HIVES.map((h) => (
              <div key={h.tag} className="hive-chip">
                <div className="hive-chip__tag">HIVE-{h.tag}</div>
                <div className="hive-chip__fob">FoB {h.fob}</div>
                <div className="hive-chip__note">{h.note}</div>
              </div>
            ))}
          </div>
        </Beat>

        <Beat id="result" kicker="The result" title="Why I trust the worse score">
          <p>
            The bars show error as <strong>MAE</strong>, which is just the average number of frames
            each prediction is off by. Lower is better. Read the top two together: when the model has
            heard the hive it misses by about three frames, and when it has not, its error more than
            triples.
          </p>
          <div className="bars">
            <MaeBar label="same hive (model has heard it)" mae={HEADLINE.sameHiveMae} ci={HEADLINE.sameHiveCi} tone="good" />
            <MaeBar label="held-out hive (never heard it)" mae={HEADLINE.lohoMae} ci={HEADLINE.lohoCi} tone="bad" />
            <MaeBar label="lazy guess (always the average)" mae={HEADLINE.meanBaselineMae} tone="mute" />
          </div>
          <p className="aside">
            The held-out error is almost as bad as a lazy model that always answers with the average
            count. It beats that lazy guess by only about {HEADLINE.skillLohoPct}%, so on a hive it has
            never heard, this model has basically no skill yet. With only {HEADLINE.nTest} test days the
            range around each number is wide, so I trust the gap between the two bars, not any single
            value. I would rather show you that gap than hand you the one number that happens to look
            best.
          </p>
        </Beat>

        <Beat id="temp" kicker="The temperature sensor" title="Does the temperature sensor actually help?">
          <p>
            Each hive also logs its own temperature and humidity. Adding those looks like free
            accuracy, right up until you hold the hive out. Temperature is the best predictor when the
            model has already seen the hive, and the worst when it has not, because the temperature
            inside a box tracks which hive and which season you are in more than it tracks how many
            bees are home. The rows below all use the same held-out days.
          </p>
          <div className="ablation">
            <div className="ablation__head">
              <span>Features</span><span>same hive</span><span>held-out</span>
            </div>
            {ABLATION.map((row) => (
              <div key={row.feature} className="ablation__row">
                <span>{row.label}</span>
                <span className="num good">{row.sameHive.toFixed(2)}</span>
                <span className="num bad">{row.loho.toFixed(2)}</span>
              </div>
            ))}
          </div>
          <p className="aside">
            So the sensor is not helping the model understand bees. It is handing it another way to
            recognise the hive, which is the same leak wearing a different hat. The paper that inspired
            this listed temperature as open work to try, so I tried it, and that is what came out.
          </p>
        </Beat>

        <Beat id="scorecard" kicker="How well it works" title="How successful is it, in plain numbers?">
          <p>
            Here is the honest scorecard. I score everything against a lazy model that ignores the
            audio and always answers with the average count, then ask how much the real model beats
            it, and by how far off it is as a percentage of the true number (MAPE).
          </p>
          <div className="stat-row">
            <Stat value={`+${HEADLINE.skillSamePct}%`} label="better than the lazy guess, on a hive it has heard" tone="good" />
            <Stat value={`${HEADLINE.sameHiveMape}%`} label="average error on that hive (off by about 3 of 16 frames)" tone="good" />
            <Stat value={`+${HEADLINE.skillLohoPct}%`} label="better than the lazy guess on a brand-new hive: basically none" tone="bad" />
            <Stat value={`${HEADLINE.lohoMape}%`} label="average error on a brand-new hive: it is guessing" tone="bad" />
          </div>
          <p>
            So how successful is it? On a hive it has already heard, quite: it cuts the error by{' '}
            {HEADLINE.skillSamePct}% and lands within about {HEADLINE.sameHiveMape}% of the real count.
            On a hive it has never heard, it fails, and that is the finding I actually care about. The
            same model, asked to generalise to a new colony, is off by {HEADLINE.lohoMape}% and barely
            beats a coin-flip average. A model that only works on hives it has already met is not
            listening to bees. It is recognising rooms.
          </p>

          <h3 className="beat__sub">How I kept it from overfitting or overtraining</h3>
          <p>
            The {HEADLINE.overfitGapPct}% gap between the two scores above is itself the overfitting
            readout: it measures exactly how much the model was leaning on hive identity. Four things
            keep that number from being even larger, and keep the same-hive score from being a fluke.
          </p>
          <ul className="decisions">
            <li>I average every hive's clips down to one row per day before training. That turns roughly 176 near-identical recordings into about 33 real data points, so the model can't pad its confidence by seeing the same afternoon a hundred times.</li>
            <li>I use Ridge regression, which adds a penalty on large weights. With this few labels, that penalty stops the model from chasing noise in the 65 audio numbers and blowing up on anything new.</li>
            <li>Leave-one-hive-out is the overfitting test, not an afterthought. A model that memorised the training hives scores well same-hive and collapses held-out, which is exactly the collapse you see, so the protocol catches overtraining instead of hiding it.</li>
            <li>Every score carries a bootstrap range and its sample size. With 9 test days I never quote a single number as if it were solid.</li>
          </ul>

          <h3 className="beat__sub">How I would push the held-out number up</h3>
          <ul className="decisions">
            <li>Train on more hives. Right now the model learns from only two, so it has almost no sense of how different colonies sound. More downloaded chunks means more hives and a real chance to generalise.</li>
            <li>Use audio that spans more inspections, so a hive's count actually moves across the clips. That gives more than 9 test points and lets me measure whether the model tracks change, not just level.</li>
            <li>Swap the raw log-mel features for Perch 2.0 embeddings. It was trained on animal sound, so it starts out already knowing what an insect drone is, instead of learning it from 43 hours.</li>
          </ul>
        </Beat>

        <Beat id="decisions" kicker="What I skipped" title="What I left out, and why">
          <ul className="decisions">
            <li>I skipped deep learning. The best published model here is a 3D convolutional network run on modulation spectrograms, but for a small, reproducible slice a plain linear model with honest error bars is a more useful ceiling to show you.</li>
            <li>I did not train on all 3,000 hours. I used one downloaded chunk, capped near {DATASET.hoursUsed} hours, because the point I care about is the method, not the scale.</li>
            <li>If I add an audio foundation model later, I will reach for Perch 2.0, which was trained on animal sounds including insects, rather than a model trained on speech or general audio.</li>
            <li>Every number carries its sample size and its error range. When the sample is tiny, I say so on the page instead of hiding it.</li>
          </ul>
        </Beat>

        <Beat id="stack" kicker="The tools" title="The tools I used, and what each one does">
          <div className="stack-grid">
            {STACK.map((s) => (
              <div key={s.layer} className="stack-card">
                <div className="stack-card__layer">{s.layer}</div>
                <div className="stack-card__choice">{s.choice}</div>
                <div className="stack-card__why">{s.why}</div>
              </div>
            ))}
          </div>
        </Beat>

        <footer className="page-foot">
          <p>
            The data is UrBAN (Abdollahi et al., <i>Scientific Data</i> 2025), used under CC BY 4.0.
            Every number here is real output from the code in this repo, run on a small slice of that
            data. Read it as a demonstration of how to test a model honestly, not as a benchmark or a
            product.
          </p>
        </footer>
      </main>
    </>
  )
}
