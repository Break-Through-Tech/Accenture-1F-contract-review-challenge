
# Contract Risk Evaluation Rubric

## Overview

This rubric provides a consistent framework for evaluating the potential risk of contractual clauses in the Accenture 1F Contract Review Challenge.

Each clause is assigned an initial risk level of **Low**, **Medium**, or **High** based on its potential financial, legal, operational, ownership, or strategic consequences.

The initial risk assignments are intended as starting points rather than definitive classifications. Actual risk depends on the specific clause wording, scope, duration, obligations, and the party being evaluated.

## 1. Risk Level Definitions

| Risk Level | Definition | What to Look For |
| --- | --- | --- |
| **Low** | Mostly routine obligations with limited consequences. | Standard administrative requirements, narrow obligations, limited duration, easy termination, minimal financial exposure, or terms that do not significantly restrict business operations. |
| **Medium** | Meaningful obligations, restrictions, or potential costs that may be manageable depending on context. | Consent requirements, renewal obligations, licensing restrictions, minimum commitments, post-termination duties, moderate financial exposure, limits on assignment, or risks dependent on scope and duration. |
| **High** | Potentially major financial, legal, ownership, or strategic consequences. | Unlimited or substantial liability, transfer of intellectual property ownership, non-compete or exclusivity restrictions, major termination consequences, change-of-control triggers, significant penalties, or broad obligations that restrict future business decisions. |

## 2. Initial Risk Assignments by Clause Category

The following table outlines preliminary risk levels for selected CUAD clause categories.

| Category | Initial Risk | Justification |
| --- | --- | --- |
| Cap On Liability | Medium | Controls maximum financial exposure. Risk depends on how high the cap is, what claims it applies to, and any exceptions. |
| Uncapped Liability | High | May expose a party to unlimited financial responsibility for certain claims or breaches. |
| Insurance | Medium | Requires specific insurance coverage and can create cost and compliance obligations. |
| Liquidated Damages | High | Creates a predetermined financial consequence if a specified breach or event occurs. |
| Warranty Duration | Medium | Longer warranty periods can extend responsibility and increase potential costs after delivery or performance. |
| Anti-Assignment | Medium | Restricts the ability to transfer contractual rights or obligations, which can limit future flexibility. |
| Change Of Control | High | An acquisition, merger, or ownership change may trigger termination, consent requirements, repayment, or loss of rights. |
| Renewal Term | Medium | Can extend contractual obligations beyond the initial term, especially if renewal happens automatically. |
| Notice Period To Terminate Renewal | Medium | Missing the notice deadline may lock a party into another contract term. |
| Termination For Convenience | Medium | Allows termination without breach; risk depends on which party holds the right and what costs or obligations follow. |
| Post-Termination Services | Medium | Creates obligations that continue after the agreement ends and may require additional resources or costs. |
| License Grant | Medium | Risk depends on the scope, duration, exclusivity, territory, sublicensing, and other rights granted. |
| IP Ownership Assignment | High | Transfers ownership of intellectual property rather than merely granting permission to use it. |
| Non-Transferable License | Medium | Limits the ability to transfer licensed rights, which may reduce flexibility during partnerships, restructuring, or acquisition. |
| Non-Compete | High | Can significantly restrict where, how, or with whom a party may compete in the future. |
| Exclusivity | High | May prevent a party from working with other customers, suppliers, distributors, or partners. |
| Minimum Commitment | Medium | Requires a minimum level of spending, purchasing, sales, or performance regardless of changing business needs. |
| Revenue / Profit Sharing | Medium | Creates ongoing financial obligations tied to revenue or profit and may affect margins. |
| Audit Rights | Medium | Can create compliance, reporting, access, and operational burdens depending on how broad the audit rights are. |

## 3. Risk Evaluation Guidelines

When evaluating an individual clause:

1. **Identify the clause category.** Determine which CUAD category best matches the clause.
2. **Assign an initial risk level.** Reference the category's preliminary risk assignment in the table above.
3. **Examine the contractual context.** Consider financial exposure, scope, duration, affected parties, restrictions, and potential consequences.
4. **Adjust the risk level if necessary.** A clause may have a higher or lower risk than its category's initial assignment.
5. **Document the justification.** Explain which specific terms or conditions support the final risk level.

### Additional Considerations

- **Financial exposure:** Does the clause create substantial costs, penalties, or liabilities?
- **Operational flexibility:** Does it restrict future business decisions or activities?
- **Duration:** How long do the obligations or restrictions remain in effect?
- **Ownership and control:** Does it transfer intellectual property or affect ownership rights?
- **Termination consequences:** Could the clause result in significant costs or loss of rights?
- **Party perspective:** Which party bears the obligation or receives the benefit?

## 4. Intended Use

This rubric is designed to support:

- Consistent preliminary contract risk classification.
- Development and evaluation of future risk-scoring approaches.
- Comparison of clause-level risk predictions with human-reviewed assessments.
- Prioritization of clauses that may require further review.

The corresponding `risk_categories.csv` file contains the category-level assignments in a structured format for use in Python and other data-processing workflows.

## 5. Limitations

- Risk levels are preliminary and should not be treated as legal conclusions.
- Category-level assignments do not automatically determine the risk of individual clauses.
- Actual risk may vary depending on contract context, jurisdiction, and the interests of the party being evaluated.
- CUAD provides clause annotations, not validated Low/Medium/High risk labels.
- Human review is recommended before using risk scores to support consequential legal or business decisions.
