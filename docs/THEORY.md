# Method theory

Let (p) be the clear-condition token distribution and (q) its registered reference distribution for the same question, image, and generation prefix. For candidate support (S), VCD forms

\[
z_i=(1+\alpha)\log p_i-\alpha\log q_i, \qquad m_i=\operatorname{softmax}_{i\in S}(z_i).
\]

The registered defaults are \(\alpha=1\), \(\beta=0.1\); the candidate mask retains tokens satisfying \(p_i\geq \beta\max_j p_j\). For a candidate event (A\subset S) and complement (B\), the event log-odds shift is

\[
\log O_m(A)-\log O_p(A)=
[\log O_{p_S}(A)-\log O_p(A)]
+\alpha[\log O_{p_S}(A)-\log O_{q_S}(A)]
+\alpha[D_{1+\alpha}(p_S(\cdot|A)\|q_S(\cdot|A))-D_{1+\alpha}(p_S(\cdot|B)\|q_S(\cdot|B))].
\]

The terms measure support masking, reference preference between event and complement, and within-group distribution differences. This identity follows by factorizing each event sum into its group probability and conditional token distribution.

For fixed (p,q,S), the effect of changing contrast weight is

\[
\frac{d m_\alpha(A)}{d\alpha}=\operatorname{Cov}_{m_\alpha}(1_A,\log p-\log q).
\]

The effect depends on the candidate event and prefix. A change in one marker's probability does not determine the change in other marker forms.

For an autoregressive sequence (y), the normalized sequence log-probability difference is the sum of local contrasts and prefix-dependent normalizers:

\[
\log M(y)-\log P(y)=\sum_t\alpha_t\log\frac{p_t(y_t|y_{<t})}{q_t(y_t|y_{<t})}-\sum_t\log Z_t(y_{<t}).
\]

For a finite set of complete candidate sequences, with (D(y)=\log M(y)-\log P(y)), event odds satisfy

\[
\log O_M(A)-\log O_P(A)=\log E_{P(\cdot|A)}e^{D(y)}-\log E_{P(\cdot|B)}e^{D(y)}.
\]

The equations apply to the recorded token support, prefixes, masks, and termination tokens.

## Registered decoders

- **VCD** contrasts clear-image logits with same-image noisy-reference logits using the event score above.
- **M3ID** uses a text-only visual reference and confidence gate. If the clear-condition maximum token probability is below 0.3, its time-varying weight is \(\alpha_t=\exp(0.02(t+o))-1\); otherwise it is zero. Offset (o) is the corresponding task prompt token count.
- **SID** forms reference logits through the registered native attention/visual intervention, then applies the configured contrast and candidate mask.
- **DoLa** chooses at each prefix the registered early layer with greatest Jensen–Shannon divergence from the final layer. It combines final and selected-layer log probabilities over the retained candidate support.
- **DeCo** creates support using `top_k=20` and `top_p=0.9`, selects the early layer with highest maximum probability on that support, then adds `alpha=0.6` times its score to final logits.

## Instruction-preserving contrast

Let (g) be guided clear-image log probability, (c) the unguided clear-image distribution, and (r) the unguided visual-reference distribution. The registered instruction-preserving score is

\[
s=g+\alpha(c-r).
\]

The visual increment is independent of the abstention instruction while the guided language contribution remains. The corresponding normalized distribution also solves the KL-regularized objective

\[
\max_\pi \; \alpha E_\pi[c-r]-KL(\pi\|G),
\]

for positive base distribution (G\), yielding \(\pi\propto G e^{\alpha(c-r)}\). Instruction-preserving M3ID computes its confidence gate and token offset from the unguided clear prompt.

## Visual CDA

CDA uses four scalar entropies of the corresponding prior/context logit distributions: image-present prior/context entropies H_p and H_c, and their empty-image counterparts H_{np} and H_{nc}:

\[
r_p=\max(H_p-H_{np},0)/H_{np},\qquad r_c=\max(H_c-H_{nc},0)/H_{nc},
\]
\[
w_p=r_p^2/(r_p+r_c),\qquad w_c=r_c^2/(r_p+r_c),\qquad w_a=1-w_p-w_c.
\]

The prior-text, context-image, and abstention-image logit vectors are combined with these step-specific weights. A zero denominator uses the registered continuous extension \((w_p,w_c,w_a)=(0,0,1)\); undefined inputs are recorded by the implementation. Equations and signed residual behavior are implemented in `src/kdm/cda.py`.

## Evidence interpretation

The equations above define the analytical quantities. Supplementary measurement requires matched prefixes, four guided/unguided clear/reference distributions, token support, layer choices, weights, and candidate scores. This measurement remains future work. The completed empirical results cover generated-response behavior, primary food-name scores, abstention, and condition-aligned method comparisons. Source literature and citation details are in `docs/REFERENCES.md`.
