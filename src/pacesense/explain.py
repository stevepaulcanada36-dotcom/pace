"""Plain-language explanations shown in the dashboard, and the list of every function it offers.

Each explanation says whether that part comes from Muijlwijk et al. (IUI '24) or was added by
PaceSense, so a reader can always tell the paper's system from the extensions.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Explanation:
    title: str
    text: str
    paper: str

    def tooltip(self) -> str:
        """Short form for a widget's help tooltip."""
        return f"{self.text}\n\n{self.paper}"


HELP: dict[str, Explanation] = {
    "weights": Explanation(
        "Race weights",
        "Tell the model how representative each past race is of what the runner can do now. Lower a race that "
        "went wrong for reasons that no longer apply, such as an injury or a hot day.",
        "From the paper: this is the interaction it tested. Coaches in the interactive condition had a slider "
        "under each race, from 0 to 1, starting at 0.5.",
    ),
    "equal": Explanation(
        "Equal weights",
        "With every race at 0.5, each race counts the same. That is the model on its own, before any coach input.",
        "From the paper: this is what coaches in the non-interactive condition saw.",
    ),
    "prediction": Explanation(
        "How the prediction is made",
        "For each past race, the model finds runners in the database who ran a very similar marathon (similar "
        "time, pacing and age) and later ran a personal best. How much they improved gives one prediction per "
        "race; the race weights combine them into one finish time.",
        "From the paper: case-based reasoning (Smyth and Cunningham, 2017), extended to use several previous "
        "races (Section 3.1).",
    ),
    "target": Explanation(
        "A target, not an average",
        "The prediction is a challenging yet realistic target: what the runner could run on a good day. Most "
        "actual results come in a little slower, which is expected.",
        "From the paper: coaches were asked for a 'challenging yet realistic' finish time, and the model "
        "predicts the same thing.",
    ),
    "range": Explanation(
        "Likely range",
        "About 8 in 10 similar runners finished inside this range. A narrow range means similar runners agree; "
        "a wide one means the prediction is less certain.",
        "Added by PaceSense. Checked on races the model had not seen: the actual result fell inside the range "
        "for about 80% of runners.",
    ),
    "cases": Explanation(
        "Similar runners used",
        "How many past runners the prediction is built from. Fewer runners usually means a wider range.",
        "Added by PaceSense, to show how much evidence sits behind the number.",
    ),
    "pacing": Explanation(
        "Reading the pacing plan",
        "Pace is minutes per kilometre (or per mile, in Settings). Each interval's time is its pace multiplied "
        "by its distance, and the interval times add up to the finish time.",
        "From the paper: the model recommends a pacing profile from similar runners' best races. In the study "
        "the pacing widget was separate from the model and did not change the finish time, and few coaches "
        "used it (Sections 4.1 and 6). Editing each interval, with the finish time following, is a PaceSense "
        "addition.",
    ),
    "intervals": Explanation(
        "Why the last interval is shorter",
        "A marathon is 42.195 km: eight 5 km intervals and a final 2.195 km. The last interval takes less time "
        "even at the same pace, so compare paces, not interval times.",
        "From the paper: the race charts use the same 5 km intervals.",
    ),
    "strategies": Explanation(
        "Pacing strategies",
        "Ready-made ways to spread the effort over the race, all starting from the model's plan. The times "
        "assume every interval is run exactly as planned; they are not separate predictions.",
        "Added by PaceSense.",
    ),
    "verdict": Explanation(
        "Realistic, ambitious or cautious",
        "The plan's finish time compared with the likely range: inside is realistic, faster is ambitious, slower "
        "is cautious. A warning appears for any interval more than 3% faster than the model's plan and faster "
        "than the runner has ever run it.",
        "Added by PaceSense.",
    ),
    "history": Explanation(
        "Past races against the model's target",
        "For each past race, the model's target using only the races before it, next to the actual result.",
        "Added by PaceSense.",
    ),
    "changes": Explanation(
        "Your changes",
        "Every change is listed with what it did to the model and to the plan, so you can see how much of the "
        "final plan comes from the model and how much from you.",
        "Related to the paper: some coaches moved the model all the way to their own guess and still counted "
        "as accepting it (Table 4). Listing each change makes that visible.",
    ),
    "watch": Explanation(
        "Race-day watch",
        "Turns a saved plan into what a runner would see on a watch during the race: the target pace for the "
        "current kilometre, how far ahead or behind plan they are, the next split, and a short cue.",
        "Added by PaceSense. The paper studied planning before the race; this shows how a plan could be used "
        "during it.",
    ),
    "assistant": Explanation(
        "Ask PaceSense",
        "An AI assistant for questions about your runners and plans. It does not calculate anything itself: it "
        "asks PaceSense to run the same functions the dashboard uses and words the answer from the results, "
        "which are listed under each answer. It also checks runner details you type and fixes common mistakes, "
        "but you confirm before anything is saved.",
        "Added by PaceSense; not part of the paper. Not available in the research study, so the study stays as "
        "the paper ran it.",
    ),
    "voice": Explanation(
        "Voice input",
        "Record (or upload) what you would type. PaceSense turns it into text, fills the matching fields, checks "
        "them, and suggests a clearer version. You review everything before it is used or saved.",
        "Added by PaceSense; not part of the paper. Uses the Speech to text, Fill fields and Refine models set in "
        "Settings.",
    ),
    "lab": Explanation(
        "Model lab",
        "The same test cases go to several AI models. Each answer is scored by code against a known right answer, "
        "and the time and cost are recorded, so you can compare model quality on evidence.",
        "Added by PaceSense; not part of the paper.",
    ),
}


@dataclass(frozen=True)
class Function:
    name: str
    where: str
    what: str
    source: str        # "Paper" or "PaceSense"
    note: str = ""


FUNCTIONS: tuple[Function, ...] = (
    Function("Find a runner", "Plan a race (step 1), Runners", "Search your own runners or the race database by "
             "name or runner number.", "Paper", "Coaches looked up their own runners in the race database (Section 4.1)."),
    Function("Add a runner", "Runners", "Enter a runner's past marathons by hand, with 5 km splits if you have them.",
             "Paper", "Coaches could add missing races by hand (Section 4.1)."),
    Function("Race weights", "Plan a race (step 2)", "Say how representative each past race is; the prediction "
             "updates.", "Paper", "The interactive condition (Section 3.2). Equal weights is the non-interactive condition."),
    Function("Model prediction", "Plan a race (step 3)", "A challenging yet realistic finish time from runners who "
             "ran a similar race and later ran their best.", "Paper", "Case-based reasoning model (Section 3.1)."),
    Function("Likely range", "Plan a race (step 3)", "The finish times about 8 in 10 similar runners went on to run.",
             "PaceSense", "Checked on held-out races: about 80% fall inside."),
    Function("Past races against the target", "Plan a race (step 3)", "How this runner's past results compared "
             "with the model's target each time.", "PaceSense"),
    Function("Pacing strategies", "Plan a race (step 4)", "Start the plan from even pace, a faster first half, a "
             "conservative start or a negative split.", "PaceSense", "The paper's model gave one recommended pacing profile."),
    Function("Edit the pacing plan", "Plan a race (step 4)", "Type a new pace for any interval; the finish time "
             "follows.", "PaceSense", "In the study the pacing widget did not affect the finish time (Section 6)."),
    Function("Plan check", "Plan a race (step 5)", "Marks the plan realistic, ambitious or cautious and warns about "
             "intervals pushed well past the model and the runner's past races.", "PaceSense"),
    Function("Your changes", "Plan a race", "Lists each change and what it did to the model and the plan.",
             "PaceSense", "Makes the 'steering' the paper describes in Table 4 visible."),
    Function("Save and export", "Plan a race (step 5)", "Save the plan, or download it as text or CSV for the runner.",
             "PaceSense"),
    Function("Race-day watch", "Race-day watch", "A watch screen for a saved plan: target pace, ahead or behind, "
             "next split and a cue, at any point in the race.", "PaceSense"),
    Function("Race simulation", "Race-day watch", "Play a simulated race (on plan, fast start, or late fade) and "
             "watch the screen and cues change.", "PaceSense"),
    Function("Pace band", "Race-day watch", "Download kilometre splits as a printable pace band, CSV or JSON.",
             "PaceSense"),
    Function("Ask PaceSense", "Ask PaceSense", "Chat about runners, plans and saved results; describe a runner "
             "in your own words and the assistant checks, corrects and prepares it for you to add. Every number "
             "comes from PaceSense's own functions.", "PaceSense", "Uses an OpenRouter model with a key, or an offline helper "
             "without one. It never sees the research study's data."),
    Function("Voice input", "Ask PaceSense, Runners, Plan a race (step 5), Saved plans", "Speak or upload a "
             "recording; it is transcribed, the right fields are filled in, and a clearer version is suggested for "
             "you to use or ignore.", "PaceSense", "Needs an OpenRouter key."),
    Function("Suggest a clearer version", "Ask PaceSense, Plan a race (step 5)", "A refined version of what you "
             "typed, with a warning if any number changed. Nothing changes until you choose it.", "PaceSense"),
    Function("AI models per task", "Settings", "Choose the OpenRouter model and temperature for each AI task: "
             "chat, speech to text, filling fields, refining.", "PaceSense"),
    Function("Model lab", "Model lab", "Run the same test cases through several models and compare score, speed "
             "and cost; pick the best model for each task.", "PaceSense"),
    Function("Record results", "Saved plans", "Enter the actual finish time after the race.", "PaceSense"),
    Function("Track record", "Track record", "Average error of your plans and of the model across recorded results.",
             "PaceSense"),
    Function("Settings", "Settings", "Guided steps or all controls, explanations, minutes per km or per mile, "
             "past-race overlay.", "PaceSense"),
    Function("Research study", "Research study", "The study as the paper ran it: four runners, then a questionnaire.",
             "Paper", "Sections 4.1 and 4.2. The dashboard does not change it."),
    Function("Study results", "Study results", "The paper's analyses of everyone who took part, and CSV exports.",
             "Paper", "Section 5. Needs the admin token."),
)
