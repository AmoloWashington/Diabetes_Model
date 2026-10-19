# Getting expert feedback on GlucoLab

GlucoLab gets better when people who work with diabetes, physiology or teaching tell us what is
wrong, unclear or missing. This guide is for the project owner: who to ask, what to send them,
and what to ask them to check.

Reviewers leave feedback on the site's **Give feedback** page (`/feedback`). It collects their role,
the pages they looked at, a usefulness rating and two free-text answers. It does not collect names
or contact details.

## Who to ask

| Reviewer | What they are best placed to judge |
|---|---|
| Doctor or endocrinologist | Clinical accuracy, safety wording, the CGM report, the monogenic diabetes page |
| Diabetes educator or nurse | Whether patients and families would understand the CGM report and the AI explanations |
| Lecturer in physiology, biochemistry or biophysics | The models, equations and the Math & physics page; use in teaching |
| Researcher (computational biology, bioinformatics) | Model implementations, RNA structure tools, validation |
| Students | Whether the site helps them learn; what confuses them |
| People living with diabetes, and carers | Whether the CGM page and the AI explanations are clear and respectful |
| Swahili-speaking health worker | The Swahili translation (see the checklist below) |

Three to five careful reviews are worth more than many quick ones.

## Message you can send

> Hello [name],
>
> I'm building GlucoLab, an open research and teaching tool for diabetes. It simulates how glucose,
> insulin and β-cells behave using published models, analyses continuous glucose monitor (CGM)
> data with the international consensus metrics, and explains the physics and maths behind each
> model. It is education software, not a medical device.
>
> Would you be willing to spend 15–20 minutes looking at it and telling me what is wrong, unclear or
> missing? Your view as a [doctor / educator / lecturer] would help me decide what to build next.
>
> The site: [link]
> The feedback form takes about three minutes: [link]/feedback
>
> Thank you,
> Washington

## What to ask reviewers to check

Pick the items that match the reviewer.

**Everyone**
- Is anything inaccurate, out of date or unsafe?
- Is anything confusing? Which words or charts?
- Would you use it, or recommend it? For what?

**Clinicians and diabetes educators**
- *Glucose data (CGM)*: are the ranges, targets and wording correct and appropriately cautious?
  Try **Generate 14 days** and **Explain my report**.
- *Monogenic diabetes*: is the gene table accurate, and is the "education, not diagnosis" message
  strong enough?
- Does any page sound like it is giving treatment advice?

**Lecturers and researchers**
- *Math & physics*: are the equations and their explanations correct?
- *Meal simulation*, *β-cell dynamics*, *Membrane biophysics*: are the models and their limits
  described fairly?

## Checklist for the Swahili translation

The Swahili interface is a **draft** and says so on every page. Before removing that notice,
ask a Swahili-speaking health worker to check:

- [ ] Medical terms: sukari kwenye damu, kisukari aina ya 1/2, insulini, seli-β, kiwango lengwa.
- [ ] The CGM page: range names, targets, the "Not medical advice" notice.
- [ ] The assistant page and the feedback form.
- [ ] That the wording is clear for patients and families, not only for professionals.
- [ ] Whether readers expect glucose in mmol/l or mg/dl.

The strings are in `web/src/lib/i18n.tsx` (the `SW` block). A reviewer can mark changes in a copy
of that list; they do not need to edit code.

## Reading the feedback

- If the server has `FEEDBACK_GITHUB_TOKEN` and `FEEDBACK_GITHUB_REPO` set, each submission
  becomes a GitHub issue titled `[Feedback] …` and you get the usual GitHub notification email.
  If the repository is public, these issues are public too.
- Submissions are also stored on the server. With `FEEDBACK_ADMIN_TOKEN` set, read them with:

  ```bash
  curl -H "x-admin-token: YOUR_TOKEN" https://your-site/api/feedback
  ```

  On Render's free tier this copy is lost when the service restarts, so use the GitHub option
  for anything you want to keep.

## Turning feedback into work

1. Group the feedback: errors, unclear wording, missing features.
2. Fix errors first, especially anything a clinician marks as unsafe.
3. Choose new features by how many reviewers ask for them and how much they would help
   patients, students or researchers.
4. Thank reviewers and tell them what changed because of them.
