# MLCommons™ AlgoPerf: Competition Rules

**Version:** 0.0.1 *(Last updated November 14, 2023)*

## Table of Contents <!-- omit from toc -->

- [Goal](#goal)
- [Sponsor](#sponsor)
- [Eligibility](#eligibility)
- [Competition Period](#competition-period)
- [Agreement to Official Rules](#agreement-to-official-rules)
- [How to Enter](#how-to-enter)
- [Submission Conditions](#submission-conditions)
- [Software Dependencies](#software-dependencies)
- [Scoring](#scoring)
- [Submissions](#submissions)
- [Optional](#optional)
- [Physical Review](#physical-review)
- [Notification](#notification)
- [Prizes](#prizes)
- [Prize Conditions](#prize-conditions)
- [Jurisdiction](#jurisdiction)
- [Cancellation and Modification](#cancellation-and-modification)
- [Publicity](#publicity)
- [Privacy](#privacy)
- [Official Rules and Winners List](#official-rules-and-winners-list)

## Goal

To discover new training algorithms that can train general neural networks faster. Sponsor will use an objective measuring program to allocate a score to each entry ("Submission") and determine two (2) winners (one in each ruleset), each of which will be eligible to win a prize.

## Sponsor

This Competition ("Competition") is sponsored by MLCommons (649 Mission Street, 5th Floor San Francisco, CA 94105, USA).

## Eligibility

The Competition is open to English-speaking individuals and teams (made of individuals), who are the age of majority as of the Competition start date, have internet access, a GitHub account in good standing, and can legally participate in this Competition ("Teams"). A Team may have unlimited participants, but all names must be entered. MLCommons Chairs and Sponsor's associated institutions are not eligible for prizes, but may participate. No natural person can be on multiple teams. This Competition is void wherever such competitions are prohibited. This Competition is subject to all applicable laws, including national, state, and local laws.

## Competition Period

The Competition begins at 12:01am (ET) on November 28, 2023 and ends at 11:59pm (ET) on May 28, 2024, all according to Sponsor's time clock, which decisions are final (the "Competition Period"). There are several deadlines contained within the Competition Period:

- **Intention to Submit.** You must register your Intention to Submit no later than 11:59pm ET on January 28, 2024.
- **Submission Period.** You must complete your Submission and enter it after the Intention to Submit deadline, but no later than 11:59pm ET on March 28, 2024.
- **Deadline for self-reporting results.** 11:59pm ET on May 28, 2024.

## Agreement to Official Rules

By participating, Teams agree to be fully unconditionally bound by these Rules, and you represent and warrant that you meet the eligibility requirements set forth herein. In addition, you agree to accept the decisions of Sponsor, as final and binding, and waive any right to claim ambiguity in the Competition or these Rules.

## How to Enter

There are four (4) steps to a successful submission ("Submission").

1. **Register Intent to Submit.** Registration of intent does not obligate you to enter a Submission, but you must register prior to entering your Submission. Click for the [Intent Form](https://forms.gle/K7ty8MaYdi2AxJ4N8). This is your "Team," even if you are a single person. Please note that natural persons may not be on multiple teams, but each Team may enter multiple Submissions.
2. **Develop your Submission.** Develop your Submission according to the guidelines set forth in these rules, along with the links to various necessary information. Please note that all Submissions must be entered subject to the Apache 2.0 license. In order to develop your Submission, you must:
   - *Fork the Benchmark Codebase.* Begin by creating a (public or private) GitHub repository for your contest submission. Once you submitted, this repository must be a clone of the frozen main branch of the benchmark codebase. Ensure that all elements of the original codebase remain unaltered, with the exception of the `/submission` directory.
   - *Preserve the Apache 2 License.* You must maintain the same Apache 2 License for your repository as the benchmark codebase. This means you may not change the licensing terms. Submissions that change the terms or otherwise fail to maintain the license, will be deemed ineligible submissions.
   - *Define Software Dependencies.* If your Submission will have any software dependencies, you must create a `requirements.txt` file in the `/submission` directory. This file must clearly list all software dependencies your Submission requires in order to be a valid Submission. File must be "pip readable" (the dependencies listed can be installed via the `pip install -r requirements.txt` command). You may not modify the package versions of the software dependencies used by the benchmarking codebase, including using a different version of libraries such as PyTorch or JAX from those specified in the benchmark.
3. **Submit Your Entry & Complete Your Submission Forms.** During the Submission Period, once your Submission is complete, submit it using the [Submission Form](https://forms.gle/yXQqwJ6Nm6sszPw49). The submission form must contain the URL of your submission's GitHub repository and the following agreements:
   - A signed [Contributor License Agreement (CLA) "Corporate CLA"](https://mlcommons.org/en/policies/) of MLCommons.
   - *Either* a membership in MLCommons *or* a signed [non-member test agreement](https://mlcommons.org/en/policies/).
   - A signed trademark license agreement, either the member or the non-member version, as appropriate. These license agreements are available upon request to [support@mlcommons.org](mailto:support@mlcommons.org).

   The form is sent to the working group chairs, who will process your Submission. Failure to complete the proper Submission Forms will results in disqualification of your Submission. At the close of the Submission Period, your GitHub repository must be public.

4. **Report Results.** Prior to the Deadline for self-reporting results, run your Submission on either the qualification set or the full benchmark set and report the results. You must report your scores by uploading all unmodified logs that the benchmarking codebase automatically generates in a separate `/results` directory within the `/submission` folder of your Submission's GitHub repository.

## Submission Conditions

All Submissions must meet the requirements of the terms contained in these rules, including reliance on new algorithmic or mathematical ideas and concepts, and must not use software engineering approaches in order to increase primitive operations in PyTorch, JAX, their dependencies, the operating systems, or the hardware. By entering, all Team members warrant that their Submission does not infringe any third party's rights, and that Team members have obtained all necessary permissions from all relevant third parties to submit the Submission. If, in the sole discretion of Sponsor, any Submission constitutes copyright or other intellectual property infringement, the Submission will be disqualified. Team must hold all rights through license or ownership to the entire Submission. Team members agree to indemnify Sponsor against any and all claims of infringement from any third party for any use by Sponsor of a Submission. Team members may not be: 1) represented under contract that would limit or impair Sponsor's ability to use the Submission; or 2) are under any other contractual relationship, including but not limited to guild and/or union memberships, that may prohibit them from participating fully in this Competition, or from allowing Sponsor to use royalty-free, the Submission worldwide in all media in perpetuity.

No Submission may depict any offensive or obscene subject matter as determined in Sponsor's sole discretion. No Submission shall portray Sponsor in a negative light. The Submission will be deemed to be owned equally by all team members, regardless of any agreement between the team members, which will not be honored by Sponsor. A Submission may be disqualified by Sponsor, in its sole discretion, if they violate the spirit and goodwill of the rules, including without limitation, if Sponsor determines a Submission is a slavish copy or derivative work of a third party that was previously developed. Submissions will be disqualified if they circumvent any rules, or protocols, including circumventing the tuning rules by looking up the result of an offline computation performed ahead of time; computing any form of pairwise metrics between the fixed and held-out workloads. Submission may use public APIs in JAX and PyTorch from within the submission function APIs, but may not use APIs to optimize the internals of primitive operations and/or standard dependencies to benefit any Submission.

## Software Dependencies

Submissions must use specific versions of PyTorch and JAX, provided by Sponsor. Additional dependencies may be added, provided Teams include a description of the additions and their function. Submissions can include dependencies that support new algorithmic and mathematical ideas provided they do not circumvent the intention of the benchmark in any way that changes measurement of the training speeds.

## Scoring

All otherwise qualified Submissions shall be scored. Submissions will be scored based on their required training time to reach the target performance on the validation set of each workload, using measuring techniques designed to give all Submissions equal parity. In the event that no Submission receives a score exceeding that of the [NAdamW baseline](https://github.com/mlcommons/algorithmic-efficiency/tree/dev/baselines/nadamw), no prizes will be awarded. The Teams with the highest scores will be determined to be winners ("Selected Teams"). In the event of a tie the prize money will be split equally between the winners.

## Submissions

Teams may enter as many Submissions as they like during the Submission Period and all otherwise qualified Submissions will be scored.

## Optional

Team members may join the Algorithm mailing group, located [here](https://groups.google.com/u/4/a/mlcommons.org/g/algorithms). This mailing group provides information to Teams regarding the status of the Competition.

## Physical Review

All Submission are subject to human review and testing to determine whether, in Sponsor's sole and exclusive discretion, any Submission fails to comply with the spirit of the Competition, and is thus disqualified.  Both physical review team and other judges shall be qualified to judge the Competition.

## Notification

On or about July 15, 2024, the Selected Team with the best scores as determined by Sponsor will be notified that they are potential winners of the Competition. The Selected Team will be notified by either phone or email at the sole discretion of Sponsor or Sponsor's representative. Selected Team will be required to respond (as directed) to a phone and/or e-mail notification within 72 hours of attempted notification. The failure to respond timely to the notification may result in forfeiture of the prize; and, in such case, Sponsor may choose the next highest scoring Submission from among the remaining eligible Submissions. Selected Team members will each be required to sign and return a Declaration (or affidavit, at Sponsor's option) of Eligibility and Liability/Publicity Release ("Declaration") and any other documents Sponsor or Sponsor's representative may require within 72 hours of receipt of the Declaration. Failure to timely return a signed Declaration (or failure of a Team member to return it), or any other required documents or the return of any prize notification as undeliverable will result in Prize forfeiture. National and state income taxes may apply and are the sole responsibility of the winner. All expenses not specifically stated as being included are excluded, and are the responsibility of the Selected Teams. No assignment, transfer or substitution of Prize is permitted, however, Sponsor reserves the right to substitute a prize for one of comparable or greater value should Prize become impracticable to award or unavailable for any reason.

## Prizes

There will be two prizes awarded, one per each ruleset. Prizes will be awarded in US Dollars. Prize will be awarded in cash, or as a gift card, at Sponsor's option. In the event the prize is a gift card, Team will be required to accept the terms and conditions of gift card. Prizes will be divided evenly among enumerated Team members listed as of the date of the Submission. In the event Sponsor is unable to award the prize, as outlined herein, for any reason, Sponsor may substitute a prize of equal or greater value.

- "Best Performance '*external-tuning*'": US $25,000
- "Best Performance '*self- tuning*'": US $25,000

## Prize Conditions

For all prizes, all national, state, province, and local taxes and other expenses in connection with the prize not expressly described herein as being awarded are the sole responsibility of the Selected Contestant. Selected Teams are solely responsible for any other unspecified expenses related to prize. Selected Teams cannot assign their prize to another person. No substitution of prize, provided however that Sponsor reserves the right to substitute a prize with another prize of equal or greater value. In the event of noncompliance with the foregoing requirements or if prize notification is returned as undeliverable, prize will be forfeited and, at Sponsor's discretion, an alternate Selected Teams with the next highest score will be chosen.

Competition is subject to these Official Rules. By participating, Teams agree: (i) to be bound by these complete Official Rules and the decisions of Sponsor which shall be final and binding; and (ii) to waive any right to claim ambiguity in the Competition or these Official Rules, except where prohibited by law. By participating in Competition or by accepting a prize, Selected Team agrees to release Sponsor, including its parent, subsidiary and affiliated entities together with the respective directors, employees, officers, licensees, licensors and agents, and respective advertising and promotion entities and any person or entity associated with the production, judging, or administration of the Competition (collectively, the "Releasees") from any and all liability, loss or damage arising from or in connection with awarding, receipt and/or use or misuse of prize or participation in any prize-related activities. Releases shall not be liable for: (i) telephone system, telephone or computer hardware, software or other technical or computer malfunctions, lost connections, disconnections, delays or transmission errors; (ii) data corruption, theft, destruction, unauthorized access to or alteration of entry or other materials; (iii) any injuries, losses or damages of any kind, including death, caused by the use of the prize money, or resulting from acceptance, possession or use of a prize, or from participation in the Competition; or (iv) any printing, typographical, administrative or technological errors in any materials associated with the Competition. Sponsor disclaims any liability for damage to any computer system resulting from participating in, or accessing or downloading information, including licenses and other information germane to the running of the Competition or otherwise in connection with this Competition. Sponsor reserves the right to cancel or suspend the Competition, in its sole discretion, should it receive fewer than two (2) prize money-eligible Submissions per ruleset, or receive no Submissions that have a judged score above a threshold set by the Sponsor, or due to circumstances beyond its control, including natural disasters, pandemic, computer virus, excessive cheating, or any other event that would undermine the fair play of the Competition. Submissions will not be returned and may be destroyed.

## Jurisdiction

The internal laws of the State of California in the United States of America will govern disputes regarding these Official Rules and/or this Contest. All cases and claims pertaining to this Contest must be brought in a court of competent jurisdiction in the City of San Francisco.

## Cancellation and Modification

Sponsor reserves the right, in its sole discretion, to cancel, modify or suspend the Competition should a virus, bug, computer problem, unauthorized intervention or other causes beyond Sponsor's control, corrupt the administration, security or proper play of the Competition. Sponsor reserves the right to cancel the competition should it receive fewer than two (2) prize money-eligible submissions per ruleset, or which are not above a threshold score as noted in these rules. Sponsor may prohibit an entrant Team (or a single person) from participating in the Competition or winning prize if, in its sole discretion, it determines such entrant is attempting to undermine the legitimate operation of the Competition in any way by cheating, hacking, deception, or any other unfair practices, including intention to annoy, abuse, threaten or harass any other competitors or Sponsor representatives. Any attempts to circumvent safeguards and benchmarks will result in disqualification, including the relevant IP address becoming ineligible for the entire Competition. Caution: any attempt to deliberately damage or undermine the legitimate operation of the Competition may be in violation of criminal and civil laws and will result in disqualification from participation in the contest. Should such an attempt be made, Sponsor reserves the right to seek remedies and damages (including attorney fees) to the fullest extent of the law, including criminal prosecution.

## Publicity

Except where prohibited, all entrants agree that Sponsor, its shareholders, agents and representatives, affiliates, subsidiaries, advertising, promotion and fulfillment agencies, and legal advisors are not responsible or liable for, and shall be released and held harmless from any and all losses, damages, rights, claims and actions of any kind in connection with or resulting from participation in the Contest, or acceptance of the prize, including without limitation, claims based on publicity rights, defamation, or invasion of privacy. Except where prohibited by law, Sponsor reserves the right to use the Submissions to the Competition, in whole or in part, for publicity purposes prior to, during, or after the Competition, in any media, and to use the name, likeness, hometown name, of any Contestant, including all or part of their Submission throughout the world, in perpetuity, without any compensation or prior review unless specifically prohibited by law. Except as outlined herein for winners, Teams and their members will not be paid for their Submissions or for granting Sponsor any of these rights. Should any Selected Team be unwilling or otherwise unable to provide permissions and or releases or otherwise cannot accept or receive the prize for any reason, the Selected Team with the next highest score will be chosen from the remaining entries until one who is able to meet all requirements can be selected

## Privacy

All personal information collected by Sponsor will be used for administration of the Competition. In addition, Team members may receive email correspondence from, or on behalf of Sponsor, via electronic communication relating to the Competition.  All personal information will be held on servers located in the United States. Sponsor will use reasonable commercial efforts to comply with Federal CAN-SPAM guidelines and other privacy guidelines, and US residents may receive commercial communications, which they may subsequently opt-out of receiving further advertising emails by following the opt-out instructions contained in any email communications received.

## Official Rules and Winners List

For a copy of these Official Rules or of the winner(s) of this Competition, send your request via email to [algorithms-chairs@mlcommons.org](mailto:algorithms-chairs@mlcommons.org). The Request and the request must be received within 90 days of the Competition end date. Please allow a reasonable time for a response.
