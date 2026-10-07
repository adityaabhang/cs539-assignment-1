"""Rebuild assignment.pdf and output/pdf/assignment1.pdf with ReportLab."""
from pathlib import Path
import shutil
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, Preformatted
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'pdf' / 'assignment1.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
# Portable fallbacks: standard PDF fonts if DejaVu is not installed.
FONT_PATH = Path('/usr/share/fonts/truetype/dejavu')
if (FONT_PATH / 'DejaVuSans.ttf').exists():
    for name, file in [('Body', 'DejaVuSans.ttf'), ('Bold', 'DejaVuSans-Bold.ttf'), ('Mono', 'DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(FONT_PATH / file)))
    pdfmetrics.registerFontFamily('Body', normal='Body', bold='Bold', italic='Body', boldItalic='Bold')
else:
    # ReportLab's Vera ships with the dependency on other platforms.
    import reportlab
    base = Path(reportlab.__file__).parent / 'fonts'
    for name, file in [('Body', 'Vera.ttf'), ('Bold', 'VeraBd.ttf'), ('Mono', 'Vera.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(base / file)))
    pdfmetrics.registerFontFamily('Body', normal='Body', bold='Bold', italic='Body', boldItalic='Bold')

INK = colors.HexColor('#172B3A')
BLUE = colors.HexColor('#BBDEFB')
GREEN = colors.HexColor('#C8E6C9')
PURPLE = colors.HexColor('#D1C4E9')
GOLD = colors.HexColor('#B38C50')
STYLES = getSampleStyleSheet()
STYLES.add(ParagraphStyle(name='Text', fontName='Body', fontSize=9.0, leading=12.4, textColor=INK, spaceAfter=6))
STYLES.add(ParagraphStyle(name='Tiny', parent=STYLES['Text'], fontSize=8.0, leading=10.6, spaceAfter=4))
STYLES.add(ParagraphStyle(name='TitleX', fontName='Bold', fontSize=23, leading=27, textColor=INK, spaceAfter=12))
STYLES.add(ParagraphStyle(name='H2X', fontName='Bold', fontSize=11.5, leading=14, textColor=INK, spaceBefore=9, spaceAfter=5))
STYLES.add(ParagraphStyle(name='Label', fontName='Bold', fontSize=9, leading=12, textColor=INK, spaceAfter=9))
STYLES.add(ParagraphStyle(name='CodeX', fontName='Mono', fontSize=7.7, leading=10.2, textColor=INK, leftIndent=9, spaceBefore=4, spaceAfter=8, backColor=colors.HexColor('#F0F3F5'), borderPadding=8))
STYLES.add(ParagraphStyle(name='Eq', fontName='Body', fontSize=10.7, leading=17, textColor=INK, alignment=1, spaceBefore=7, spaceAfter=11))
story = []


def para(text, style='Text'):
    story.append(Paragraph(text, STYLES[style]))


def title(kicker, heading):
    para(kicker.upper(), 'Label')
    para(heading, 'TitleX')


def h(text):
    para(text, 'H2X')


def code(text):
    story.append(Preformatted(text, STYLES['CodeX']))


def eq(text):
    para(text, 'Eq')


def table(headers, rows, widths, color=BLUE):
    data = [[Paragraph(str(x), STYLES['Tiny']) for x in row] for row in [headers, *rows]]
    t = Table(data, colWidths=widths, hAlign='LEFT', repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),color), ('TEXTCOLOR',(0,0),(-1,-1),INK),
                          ('VALIGN',(0,0),(-1,-1),'TOP'), ('LEFTPADDING',(0,0),(-1,-1),8),
                          ('RIGHTPADDING',(0,0),(-1,-1),8), ('TOPPADDING',(0,0),(-1,-1),6),
                          ('BOTTOMPADDING',(0,0),(-1,-1),4), ('LINEBELOW',(0,0),(-1,0),.7,INK),
                          ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F3F5F6'),colors.white]),
                          ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#D6DFE4'))]))
    story.append(t)
    story.append(Spacer(1,8))


def callout(heading, text, color=GREEN):
    t = Table([[Paragraph('<b>'+heading+'</b><br/>'+text, STYLES['Text'])]], colWidths=[504])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),color), ('BOX',(0,0),(-1,-1),.4,colors.HexColor('#84939B')),
                          ('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12),
                          ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    story.append(t)
    story.append(Spacer(1,8))


def page():
    story.append(PageBreak())


# Page 1
title('CS59300-ILR | Fall 2026 | Assignment 1', 'From demonstrations<br/>to interactive imitation')
para('Collect data. Learn a policy. Diagnose distribution shift.', 'H2X')
para('In this assignment you will build and compare behavior cloning (BC) and DAgger across control and combinatorial decision making. The environment adapters, interactive interfaces, teachers, network, storage, and experiment runner are supplied. Your implementation work is concentrated in five functions.')
callout('Scope and workload', '<b>50 demonstrations per Lander variant and 10 Chess games; 110 total.</b> Budget approximately 3-6 hours of active work. Chess games record up to 20 White moves and then a bot takes over. CPU training is sufficient. The time budget is an estimate and depends on game familiarity; no target win rate is required.', BLUE)
table(['Variant', 'One demonstration', 'Main learning challenge'], [
    ['Lunar Lander: discrete', 'An episode; at most 600 physics steps / 300 decisions', 'Four action classes; compounding control errors'],
    ['Lunar Lander: continuous', 'Same episode cap and decision frequency', 'Two graded controls; regression and actuator dead zones'],
    ['Chess', '20 White moves; automatic Black replies', 'Sparse coverage of structured positions and move choices'],
], [113,192,199])
h('Your deliverables')
para('<b>Code:</b> complete <font name="Mono">student/algorithms.py</font>. <b>Evidence:</b> three frozen demonstration banks, experiment manifests, raw results, and plots. <b>Report:</b> 4-6 pages analyzing quantity, action representation, and DAgger. Include a reproduction appendix outside the page limit.')
table(['Suggested active-time budget', 'Target'], [
    ['Setup, reading, and practice', '20-40 minutes'],
    ['Collect three banks (50 + 50 + 10 demonstrations)', '20-30 minutes'],
    ['Implement, test, and run experiments', '90-150 minutes'],
    ['Analyze results and write report', '60-120 minutes']
], [370,134], GREEN)
para('Learning target: explain why low supervised loss does not guarantee high rollout return, why feedback on learner-visited states can help, and why neither more data nor DAgger guarantees improvement in every finite experiment.', 'Tiny')
page()

# Page 2
title('01 | Data collection', 'A demonstration is a trajectory')
para('Practice into a separate folder, then keep all completed collection episodes, including failures. Do not cherry-pick successful landings or winning openings. You play White in Chess. Short early endings still count as one demonstration; report their frequency.')
code('ilr collect --domain lander-discrete --data data/lander-discrete\nilr collect --domain lander-continuous --data data/lander-continuous\nilr collect --domain chess --data data/chess')
para('Lander commands default to 50 episodes each; Chess defaults to 10 games. Press <b>Enter</b> to begin an episode and <b>Esc</b> to stop. Repeating the same command resumes by skipping saved seed-numbered files. Unfinished prefixes are discarded. A completed prefix is saved before bot takeover begins.')
table(['Variant', 'Interactive controls'], [
    ['Lander discrete', 'Up: main engine. Left/right: corresponding jet. No key: coast. Main has priority.'],
    ['Lander continuous', 'Hold the mouse in the control pad: vertical main throttle, horizontal side throttle. W/S and A/D ramp controls; Space cuts main.'],
    ['Chess', 'Click source and destination. Q/R/B/N chooses promotion. Only legal moves are accepted.'],
], [112,392])
h('Human labels and assistance')
para('<b>H</b> accepts one teacher suggestion and marks that row <font name="Mono">assisted</font>. Report the assisted fraction. The headless <font name="Mono">--mode teacher</font> option is for infrastructure smoke tests; these synthetic banks do not satisfy human collection. Do not use <font name="Mono">--allow-teacher</font> for the required experiment.')
h('Prefix versus continuation')
para('Chess records up to 20 <b>student-side moves</b>, not 20 total plies. Black uses Stockfish with seeded occasional random opening moves to provide variation. After the prefix, Stockfish takes over. Games stop after at most 160 plies; natural endings can occur earlier.')
callout('Do not learn from the takeover suffix', 'Training uses only the recorded human prefix. The eventual win/draw/loss measures a <b>hybrid policy: learned or human prefix + bot continuation</b>. It is useful but does not measure unaided full-game skill.', PURPLE)
h('Freeze the bank before making subsets')
para('Complete and freeze each bank first: 50 files per Lander variant and 10 files for Chess. The loader shuffles complete episodes once with a fixed selection seed. Lander uses nested subsets of 5, 10, and 50 demonstrations; Chess uses nested subsets of 5 and 10 games. Do not split adjacent transitions from the same episode between training and evaluation. Later additions to a bank change its permutation.')
page()

# Page 3
title('02 | Representation', 'What does the learner observe?')
table(['Variant', 'Input x', 'Output and restrictions'], [
    ['Lander discrete', '8 numbers: position, velocity, angle, angular velocity, and two leg-contact flags', '4 logits; select one engine action'],
    ['Lander continuous', 'The same 8 observation components', '2 tanh outputs in [-1,1]; main and side controls'],
    ['Chess', '1,032 features from the fully visible position and counters', '2,034 logits; mask to legal chess moves'],
], [101,217,186], PURPLE)
para('The supplied network is an MLP with two 64-unit ReLU hidden layers. Board features are 16 x 64 planes plus 8 global values. The fixed output layout contains 1,968 geometric moves/promotions and 66 unused slots, which are always masked out. Two input planes and one global input are fixed zeros. Only legal Chess moves can be selected. See <font name="Mono">encode_board</font> and <font name="Mono">Policy</font> for the exact layout.')
para('Chess exposes the current board, but its feature vector omits full repetition history even though the rules engine tracks it. The fixed representation limits the conclusions you can draw.')
h('Inspect one episode and measure its size')
code('import numpy as np\nwith np.load("data/chess/episode_00000.npz", allow_pickle=False) as z:\n    print(z["obs"].shape, z["actions"].shape, z["masks"].shape)\n    print(z["sources"], z["metadata"])')
para('Fields include observations, teacher/demonstrator targets, masks, executed actions, phase, source, and metadata. For 20 board decisions, observations alone use 20 x 1,032 x 4 = <b>82,560 bytes</b>; Boolean masks add <b>40,680 bytes</b> before compression. Report actual bank sizes and decision counts. More demonstrations do not imply equal amounts of data across variants.', 'Tiny')
para('Chess increases combinatorial and action-selection difficulty. Do not claim its mathematical state-space cardinality exceeds that of a continuous system merely because its input vector is larger.', 'Tiny')
page()

# Page 4
title('03 | Implement behavior cloning | 30 points', 'Fit actions from demonstrations')
para('Let D contain N labeled decisions (x<sub>i</sub>, a<sub>i</sub><super>*</super>, m<sub>i</sub>), where m identifies the action choices available at collection time. BC minimizes a supervised loss over this fixed dataset. For Lander, the provided driver fits normalization statistics using only the selected training subset. Board features already have fixed scales.')
h('1. bc_loss(predictions, targets, masks, continuous) [15]')
para('<b>Continuous:</b> use mean squared error across the batch and both action dimensions. Predictions and targets have shape [B, 2]. The returned loss is a scalar tensor supporting backpropagation.')
eq('L<sub>cont</sub> = (1 / 2B) ∑<sub>i=1</sub><super>B</super> ∑<sub>j=1</sub><super>2</super> (π<sub>θ</sub>(x<sub>i</sub>)<sub>j</sub> - a<sub>ij</sub><super>*</super>)<super>2</super>')
para('<b>Discrete:</b> predictions are raw logits [B, A], targets are class indices [B], and masks are Boolean [B, A]. Set disallowed logits to negative infinity <b>before</b> cross-entropy. Do not apply softmax first. Every provided training mask contains its target.')
eq('L<sub>disc</sub> = -(1 / B) ∑<sub>i=1</sub><super>B</super> log [ exp(z<sub>i,a*</sub>) / ∑<sub>a ∈ A(m_i)</sub> exp(z<sub>i,a</sub>) ]')
para('Explain why masking only the selected action at inference is insufficient. A logit for an unavailable move must not consume probability mass in the training objective.')
h('2. train_bc(policy, data, *, epochs, batch_size, lr, seed) [15]')
para('Train with shuffled minibatch Adam using <font name="Mono">bc_loss</font>. Set training mode, clear gradients, backpropagate, and step the optimizer. Include the final smaller batch. Shuffle reproducibly using a seeded Torch generator. Return one sample-weighted mean loss per epoch. Do not reset weights or normalizer when called again: DAgger warm-starts the same policy.')
code('python -m pytest tests/test_algorithms.py -q\nilr train --domain lander-discrete --data data/lander-discrete \\\n  --demos 5 --out runs/debug-bc\nilr evaluate --checkpoint runs/debug-bc/policy.pt \\\n  --split validation --episodes 5 --out runs/debug-validation.json')
callout('Supervised fit and rollout performance differ', 'A small error changes the next observation, which can lead to further errors outside the demonstration distribution. Conversely, disagreement with a teacher can still produce a reasonable action. Compare losses, teacher disagreement, and actual rollouts; do not treat them as interchangeable. [1,2]', GREEN)
para('Continuous nuance: main-engine outputs below zero are off; nonnegative outputs correspond to 50-100% thrust. Side thrust has a dead zone. Equal differences in numerical action need not produce equal changes in physical behavior. [3]', 'Tiny')
page()

# Page 5
title('04 | Implement DAgger | 30 points', 'Label the states the learner visits')
para('Initialize with BC-50 for Lander or BC-10 for Chess. In each round, roll out a mixture of the current learner and a supplied teacher, obtain the teacher action at every visited input, aggregate the new labels with all previous data, and train again. The supplied runner handles checkpoints, fresh seeds, and evaluation. [2]')
h('3. beta_schedule(round_index) [5]')
para('Return the probability of <b>executing</b> the teacher action. Indices are zero-based: β<sub>i</sub> = 0.5<super>i+1</super>, giving 0.5, 0.25, and 0.125. Reject a negative index. This changes who acts; the teacher remains the source of every new training target.')
h('4. collect_dagger_episode(env, policy, beta, rng) [20]')
para('The environment has already been reset. Repeat the following until <font name="Mono">env.prefix_done</font>; do not reset it inside your function.')
table(['Order', 'Required operation'], [
    ['1', 'Read obs, mask = env.observe(); preserve this pre-action input.'],
    ['2', 'Query env.expert_action() and predict policy.act(obs, mask).'],
    ['3', 'With probability beta execute the teacher; otherwise execute the learner.'],
    ['4', 'Append sample(obs, mask, teacher_label, executed, env.phase, "dagger").'],
    ['5', 'Call env.step(executed), then repeat from the newly visited input.']
], [49,455], GREEN)
para('Return the row list. Do not call <font name="Mono">env.finish()</font>, append the bot suffix, store the learner action as the target, query the teacher after stepping.')
h('5. aggregate_dataset(old, new) [5]')
para('Return a fresh <font name="Mono">Batch</font> by concatenating observations, targets, and masks along axis 0. Keep all earlier rows, including those from every previous DAgger round. Support discrete targets [N] and continuous targets [N, 2]. Do not mutate the inputs.')
eq('D<sub>i+1</sub> = D<sub>i</sub> ∪ { (x, π<sub>E</sub>(x), m) : x visited in round i }')
callout('Queryable teachers have limitations', 'Lander uses the Gymnasium heuristic; Chess uses a limited-node Stockfish engine. These are practical label providers, not optimal or omniscient oracles. [3,4]', PURPLE)
para('Human BC labels and bot DAgger labels can differ systematically. A DAgger improvement may reflect better state coverage, different teacher quality, or both. Discuss this confound rather than attributing every gain to the algorithm alone.', 'Tiny')
page()

# Page 6
title('05 | Required experiment matrix', 'Compare data quantity and feedback')
para('Run the default sweep once per variant after completing the five functions. Use the same frozen bank, architecture, and evaluation settings throughout. One training seed is required; three independent training seeds are a useful optional extension. Hyperparameter search is not required.')
code('ilr sweep --domain lander-discrete --data data/lander-discrete \\\n  --out runs/lander-discrete\nilr sweep --domain lander-continuous --data data/lander-continuous \\\n  --out runs/lander-continuous\nilr sweep --domain chess --data data/chess --out runs/chess')
table(['Condition', 'Data and updates', 'Evaluate'], [
    ['BC subsets', 'Lander: 5 / 10 / 50 demonstrations. Chess: 5 / 10 games. Nested human subsets; 12 epochs per model.', 'Each model on the same 20 held-out seeds'],
    ['DAgger rounds 1 / 2 / 3', 'Start BC-50 for Lander or BC-10 for Chess; 5 new prefixes per round; beta = .5 / .25 / .125; aggregate and train 12 epochs', 'Learner-only prefix after each round'],
    ['BC extra-training control', 'Start from the same BC-50 (Lander) or BC-10 (Chess); original data only; approximately match the additional DAgger optimizer updates', 'Same held-out evaluation protocol']
], [119,250,135], BLUE)
para('Defaults: Adam learning rate 0.001, minibatch size 128, CPU execution. The control rounds up to complete BC epochs; its update count is recorded and differs by less than one BC epoch. Both methods retain weights but reset the optimizer on each training call. The control helps assess additional optimization, not extra labeling effort.')
h('Holdout discipline and uncertainty')
para('Human collection uses seeds below 10,000; validation starts at 10,000, final testing at 20,000, and DAgger at 30,000. Use validation runs for debugging or any tuning, then freeze settings before final testing. Do not tune to the final test curves. The runner reuses evaluation seeds across conditions and saves raw episode-level outcomes.')
para('Plot means with the supplied 95% episode-bootstrap intervals. Twenty episodes can yield wide or even deceptively narrow intervals for sparse win/loss outcomes. These intervals describe evaluation variation conditional on one trained model; they do not capture training-seed uncertainty. Inspect paired per-seed differences when discussing small improvements.')
h('Outputs to inspect')
para('<font name="Mono">results.csv</font> contains raw episode metrics; <font name="Mono">results.json</font> includes summaries and protocols; <font name="Mono">learning_curves.png</font> plots the main quantity and DAgger comparisons. Each BC directory has a training manifest and checkpoint. Each DAgger round also records cumulative teacher labels, aggregate dataset size, and loss.')
callout('Label budget matters', 'Report new and cumulative oracle labels, not just round count. Fifteen new Lander rollouts can contain many more labeled decisions than fifteen board prefixes. Report collection assistance separately from DAgger queries.', GREEN)
page()

# Page 7
title('06 | Evaluation and interpretation', 'Tell the right story with the metrics')
para('For Lander, define episode return G<sub>0</sub> = ∑<sub>t=0</sub><super>T-1</super> r<sub>t</sub> with γ = 1 for this benchmark. Report the empirical mean of G<sub>0</sub> and the fraction with G<sub>0</sub> ≥ 200. The teacher is never mixed into the evaluation prefix. Action repeat sums both physics-step rewards.')
table(['Domain', 'Primary performance', 'Required supporting diagnostics'], [
    ['Lander (both)', 'Mean return; fraction with return >= 200', 'Truncation fraction; continuous control MSE or discrete teacher disagreement'],
    ['Chess', 'Hybrid W/D/L and score = win + 0.5 draw', 'Material at the prefix boundary; move disagreement; capped-game fraction'],
], [92,186,226], PURPLE)
para('Material uses pawn=1, knight=bishop=3, rook=5, queen=9, from White’s perspective, measured at the prefix boundary. Chess stops at natural game endings or the 160-ply cap; report capped endings separately.')
h('Required figures')
para('<b>Figure 1:</b> three panels showing BC performance: 5/10/50 demonstrations for each Lander variant and 5/10 games for Chess.<br/><b>Figure 2:</b> three panels showing DAgger performance by round or cumulative oracle labels, with the initial BC baseline (BC-50 for Lander, BC-10 for Chess) and extra-training control.<br/><b>Figure 3:</b> at least one diagnostic plot from the raw results, such as prefix material or teacher disagreement. Keep return and game-score axes distinct.')
h('Questions your report must answer')
para('<b>Quantity and quality:</b> Which variants benefit from more demonstrations? Could coverage, correlated trajectories, longer episodes, novice improvement during collection, assistance, or label inconsistency explain a trend? Include the actual decision counts.')
para('<b>Actions and representations:</b> How does continuous MSE differ from masked classification? Explain Lander dead zones. Why do Chess openings sparsely cover useful decisions?')
para('<b>Feedback:</b> Does DAgger outperform both its initial BC baseline (BC-50 for Lander, BC-10 for Chess) and the extra-training control? If not, investigate teacher quality, weak learner coverage, mixed data weighting, and finite-sample uncertainty. What distinguishes disagreement from poor task performance?')
callout('Discuss failures thoughtfully', 'Describe one concrete failure or unexpected trend in each domain family. Explain why hybrid final outcomes may hide a weak prefix, and propose one controlled follow-up that distinguishes two plausible explanations. Negative or non-monotonic results can receive full credit when the implementation and analysis are sound.', GREEN)
page()

# Page 8
title('07 | Setup, grading, and submission', 'Finish with a reproducible result')
h('Local setup')
para('Use Python 3.12 and a visible desktop for human collection. Detailed Linux, macOS, and Windows instructions are in <font name="Mono">README.md</font>. Install Stockfish separately from its official site and set <font name="Mono">STOCKFISH_EXECUTABLE</font> if needed. Keep its version fixed. A GPU is optional.')
code('python3.12 -m venv .venv\nsource .venv/bin/activate\npython -m pip install --upgrade pip setuptools wheel\n# Linux CPU; macOS uses the default torch index instead:\npython -m pip install torch==2.8.0 \\\n  --index-url https://download.pytorch.org/whl/cpu\npython -m pip install -e ".[dev]"\nexport STOCKFISH_EXECUTABLE=/usr/games/stockfish\nilr doctor\npython -m pytest tests/test_infrastructure.py -q')
para('If Box2D must compile, install a compiler and SWIG first; the README gives platform-specific commands. The algorithm tests initially fail with <font name="Mono">NotImplementedError</font>. Implement only the five required functions for the main experiment; the supplied infrastructure tests should pass before you start.')
table(['Assessment', 'Points', 'Evidence'], [
    ['BC implementation', '30', 'Correct differentiable losses, masking, and minibatch training'],
    ['DAgger implementation', '30', 'Correct mixture, pre-action teacher labels, and cumulative aggregation'],
    ['Collection and protocol', '8', 'Three banks (50/50/10), provenance, row counts, data size, frozen nested subsets'],
    ['Experimental evidence', '12', 'Required curves, control, uncertainty, diagnostics, reproducible settings'],
    ['Interpretation', '16', 'Thoughtful trends, failures, teacher and metric limitations'],
    ['Clarity and reproducibility', '4', 'Readable report, citations, commands, manifests, versions']
], [164,48,292], GREEN)
h('Submit')
para('A 4-6 page report PDF; completed <font name="Mono">student/algorithms.py</font>; three human NPZ banks (50 discrete Lander, 50 continuous Lander, 10 Chess) and outcome sidecars; result CSV/JSON files; plots; manifests and reproduction commands. Exclude the virtual environment, Stockfish binary, practice/smoke data, caches, and instructor solutions. Keep checkpoints locally; include them only if requested by the course staff.')
h('Before submitting')
para('Verify that each Lander bank contains 50 demonstrations and the Chess bank contains 10 games; assistance is disclosed; smaller subsets are nested; bot suffixes never became training rows; evaluation used the learner alone during its prefix; and every figure identifies the measured quantity and uncertainty. The report template lists the expected structure.')
page()

# Page 9
title('08 | Reading and implementation map', 'Connect the experiment to the literature')
refs = [
    ('1. Osa et al. (2018)', 'An Algorithmic Perspective on Imitation Learning. Foundations and Trends in Robotics.', 'https://arxiv.org/abs/1811.06711', 'Read Section 2.1 for design choices: who demonstrates, what is recorded, and how policies are represented.'),
    ('2. Ross, Gordon, and Bagnell (2011)', 'A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. AISTATS.', 'https://proceedings.mlr.press/v15/ross11a.html', 'Read the motivation and DAgger algorithm. Our short experiment is not a verification of an asymptotic theorem.'),
    ('3. Farama Foundation', 'Gymnasium LunarLander-v3 documentation and source.', 'https://gymnasium.farama.org/environments/box2d/lunar_lander/', 'Version 1.2.3 supplies the simulator and heuristic. Classroom wrappers add action repeat and the short episode cap.'),
    ('4. Stockfish developers', 'Stockfish chess engine, installed separately.', 'https://stockfishchess.org/', 'Record the engine build. A fixed 1,024-node budget supports short runs but does not make the teacher optimal.')
]
for label, paper, url, note in refs:
    para('<b>'+label+'</b><br/>'+paper+' <link href="'+url+'" color="#285D83"><u>Open source</u></link><br/>'+note, 'Tiny')
h('Code map')
table(['File', 'Purpose'], [
    ['student/algorithms.py', 'The five graded functions; exact contracts in docstrings'],
    ['il_assignment/domains.py', 'Observation encoding, teachers, action restrictions, and prefix boundaries'],
    ['il_assignment/ui.py', 'Interactive controls and collection visualization'],
    ['il_assignment/data.py', 'Episode format, provenance, and nested data subsets'],
    ['il_assignment/model.py', 'Provided network and observation normalization'],
    ['il_assignment/experiments.py', 'Training, DAgger driver, evaluation, summaries, and plots'],
    ['tests/; report_template.md', 'Public correctness checks and report scaffold']
], [189,315], BLUE)
para('The instructor release contains separate worked implementations. The student archive excludes them. Third-party code is imported from its published package; no engine binary or external image is bundled. Dependency provenance is documented in <font name="Mono">THIRD_PARTY.md</font>.', 'Tiny')


def chrome(c, doc):
    c.saveState()
    w, height = doc.pagesize
    c.setFillColor(INK)
    c.rect(0, height-12, w, 12, fill=1, stroke=0)
    c.setFillColor(GOLD)
    c.rect(54, height-35, 45, 3, fill=1, stroke=0)
    c.setStrokeColor(colors.HexColor('#C8D2D8'))
    c.line(54, 43, w-54, 43)
    c.setFillColor(INK)
    c.setFont('Body', 7.7)
    c.drawString(54, 29, 'CS59300-ILR  |  Assignment 1  |  Fall 2026')
    c.drawRightString(w-54, 29, f'{doc.page}')
    c.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=(612,792), leftMargin=54, rightMargin=54,
                        topMargin=54, bottomMargin=57, title='Assignment 1: From demonstrations to interactive imitation',
                        author='CS59300-ILR | Imitation Learning for Robotics')
doc.build(story, onFirstPage=chrome, onLaterPages=chrome)
shutil.copyfile(OUT, ROOT / 'assignment.pdf')
print(OUT)
