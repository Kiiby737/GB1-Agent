"""Deterministic checks for explicit historical arrows and fitness differences.

These checks verify concrete cited pairs; they do not validate general biological
mechanisms or interpret arbitrary natural-language claims.
"""
import re
import math
from collections import defaultdict
from decimal import Decimal
from itertools import combinations

AA = 'ACDEFGHIKLMNPQRSTVWY'
POSITIONS = (39, 40, 41, 54)
ARROW = re.compile(rf'(?<![A-Z])([{AA}]{{4}})\s*(?:→|->|⇒)\s*([{AA}]{{4}})(?![A-Z])')
DELTA = re.compile(r'(?:Δ\s*(?:fitness)?|delta(?:_fitness)?|差值)\s*(?:[=:：]|为|是)\s*([+−-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', re.IGNORECASE)
CONTEXT = re.compile(rf'在\s*([{AA}]{{3}})\s*背景[^。；;\n]*?([{AA}])(39|40|41|54)([{AA}])')
TOKEN = rf'(?<![A-Z])([{AA}]{{3,4}})(?![A-Z])'
SITE = r'(?:第\s*)?(?:39|40|41|54)\s*(?:位点|位)?'
SITE_LIST = rf'{SITE}(?:\s*(?:、|，|,|和|及|与|/|\+)\s*{SITE})*'
BACKGROUND_PAIR = rf'{TOKEN}\s*(?:的?背景)?\s*[，,]?\s*(?:与|和|相对(?:于)?)\s*{TOKEN}\s*(?:的?背景)?\s*(?:相比)?\s*'
BACKGROUND_DIFFERENCES = (
    re.compile(BACKGROUND_PAIR + rf'(?P<only>仅仅|仅|只有|只)?\s*(?:在)?\s*(?P<sites>{SITE_LIST})\s*(?:上)?\s*(?:存在)?(?:不同|有差异|差异)'),
    re.compile(BACKGROUND_PAIR + rf'(?:仅|只)?\s*(?:相差|差异位点(?:为|是|[:：])|差)\s*(?P<sites>{SITE_LIST})'),
)
EFFECT = re.compile(r'正效应|正向|提高|增加|改善|有益|提升|负效应|负向|降低|下降|有害')
POSITIVE_EFFECTS = {'正效应', '正向', '提高', '增加', '改善', '有益', '提升'}
EPI_ID = re.compile(r'epistasis:\d+-\d+:[A-Z]+:[A-Z]+:[A-Z]+(?![A-Z])')
EPSILON = re.compile(r'(?:ε|epsilon(?:_in_proposed_direction|_canonical)?)\s*'
                     r'(?:[=:：≈≃~～]|约为|为|是|约)?\s*'
                     r'([+−-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?(?:nan|inf(?:inity)?))',
                     re.IGNORECASE)
PAIR_METRIC_SCOPE = re.compile(
    r'(?P<other>(?:其他|其它|跨|不同)(?:固定)?背景|'
    r'(?:KG|历史)(?:(?!当前配对|本配对|该配对|此配对)[^，,。；;\n]){0,10}?'
    r'(?:中位(?:数)?|均值|平均(?:值)?)|'
    r'(?:模型|M0(?:-AL)?)(?:的)?预测|预测(?:Δ|delta|差值|效应|Fitness))|'
    r'(?P<current>(?:当前|本|该|此)(?:配对|比较)|当前背景|'
    r'(?:当前|本轮|现在)(?:的)?(?:实测|实际|测得))', re.IGNORECASE)
PREDICTION_PREFIX = re.compile(r'(?:模型|M0(?:-AL)?).{0,6}预测|预测\s*$', re.IGNORECASE)


def measured_pair_clauses(tail, prefix=''):
    """Separate this pair's measurements from aggregate and predicted metrics.

    A scope switch affects only subsequent text: adding a background statistic
    after a false measured assertion cannot exempt that assertion from checks.
    """
    active = not bool(PREDICTION_PREFIX.search(prefix))
    for clause in re.finditer(r'[^，,]+', tail):
        text = clause.group()
        lead = re.sub(r'^\s*(?:(?:而|但|然而|不过|则|现在|目前|相反)\s*)*', '', text)
        if re.match(r'实测|测得|实际(?:Δ|差值|效应|结果)', lead):
            active = True
        cursor = 0
        for marker in PAIR_METRIC_SCOPE.finditer(text):
            if active and marker.start() > cursor:
                yield clause.start() + cursor, text[cursor:marker.start()]
            active = marker.lastgroup == 'current'
            # "不能外推当前配对有益" is a limitation, not a new assertion.
            if active and re.search(r'(?:不能|不可|无法|不应|不足以)(?:直接)?'
                                    r'(?:外推|推断|证明|认定)(?:到|至|为)?\s*$', text[:marker.start()]):
                active = False
            cursor = marker.start()
        if active and cursor < len(text):
            yield clause.start() + cursor, text[cursor:]


def pair_claim_prefix(text, arrows, index):
    arrow = arrows[index]
    prefix = re.split(r'[，,。；;\n]', text[:arrow.start()])[-1]
    if ARROW.search(prefix):
        # The previous arrow's effect words belong to that previous pair.
        qualifier = re.search(r'(?:不能将|不能把|不可将|不应将|若|如果|可能|预期)[^，,。；;\n]{0,10}$', prefix)
        prefix = qualifier.group() if qualifier else ''
    return prefix[-20:]


def epistasis_records(observed):
    """One shared four-corner calculation for KG exports and factual checks."""
    fitness = (observed if isinstance(observed, dict) else
               dict(zip(observed['variant'], observed['fitness'])))
    rows = []
    for i, j in combinations(range(4), 2):
        fixed = [k for k in range(4) if k not in (i, j)]
        groups = defaultdict(dict)
        for variant in fitness:
            background = ''.join(variant[k] for k in fixed)
            groups[background][(variant[i], variant[j])] = variant
        for background, grid in groups.items():
            for pair_i in combinations(sorted({a for a, _ in grid}), 2):
                for pair_j in combinations(sorted({b for _, b in grid}), 2):
                    a0, a1 = (('VDGV'[i], next(x for x in pair_i if x != 'VDGV'[i]))
                              if 'VDGV'[i] in pair_i else pair_i)
                    b0, b1 = (('VDGV'[j], next(x for x in pair_j if x != 'VDGV'[j]))
                              if 'VDGV'[j] in pair_j else pair_j)
                    corners = ((a0, b0), (a1, b0), (a0, b1), (a1, b1))
                    if not all(corner in grid for corner in corners):
                        continue
                    v00, v10, v01, v11 = (grid[c] for c in corners)
                    f00, f10, f01, f11 = (float(fitness[v]) for v in (v00, v10, v01, v11))
                    if not all(math.isfinite(f) for f in (f00, f10, f01, f11)):
                        raise ValueError('Epistasis requires four finite prior measured fitness values')
                    rows.append({
                        'position_a': POSITIONS[i], 'position_b': POSITIONS[j],
                        'background': background,
                        'context_id': f'context:{POSITIONS[i]}-{POSITIONS[j]}:{background}',
                        'substitution_a': f'{a0}{POSITIONS[i]}{a1}',
                        'substitution_b': f'{b0}{POSITIONS[j]}{b1}',
                        'baseline_variant': v00, 'single_a_variant': v10,
                        'single_b_variant': v01, 'double_variant': v11,
                        'f00': f00, 'f10': f10, 'f01': f01, 'f11': f11,
                        'epsilon': f11 - f10 - f01 + f00,
                    })
    return rows


def epistasis_evidence_index(observed):
    index = {}
    for row in epistasis_records(observed):
        a, b = row['substitution_a'], row['substitution_b']
        key = (f"epistasis:{row['position_a']}-{row['position_b']}:{row['background']}:"
               f'{a[0]}{a[-1]}:{b[0]}{b[-1]}')
        index[key] = row
    return index


def proposed_substitutions(reference, test):
    if (len(reference) != 4 or len(test) != 4 or not set(reference + test) <= set(AA)):
        raise ValueError('Epistasis target must use canonical four-site variant IDs')
    return tuple(f'{a}{pos}{b}' for a, b, pos in zip(reference, test, POSITIONS) if a != b)


def _epistasis_direction(record, focus, description):
    if not focus:
        return 1  # General citations use the record's canonical direction.
    direction, matched = 1, set()
    for target in focus:
        reverse = target[-1] + target[1:-1] + target[0]
        for historical in (record['substitution_a'], record['substitution_b']):
            if target == historical:
                matched.add(target)
            elif reverse == historical:
                direction *= -1
                matched.add(target)
    if len(matched) != len(focus):
        raise ValueError(f'{description}: epistasis substitutions '
                         f"{record['substitution_a']}/{record['substitution_b']} do not match "
                         f'target {list(focus)}; use evidence for the same substitution')
    return direction


def _check_epsilon(number, expected, description):
    number = str(number).replace('−', '-')
    actual = float(number)
    if not math.isfinite(actual):
        raise ValueError(f'{description}: nonfinite epsilon')
    tolerance = .5 * 10 ** Decimal(number).as_tuple().exponent + 1e-5
    if abs(actual - expected) > tolerance:
        raise ValueError(f'{description}: epsilon={expected:+.8f}, not {number}; '
                         'canonical epsilon is f11-f10-f01+f00; reversing one substitution reverses its sign')


def epistasis_reference_facts(eid, reference, test, observed, *, index=None):
    """Return authoritative citation facts without trusting LLM numeric fields."""
    index = epistasis_evidence_index(observed) if index is None else index
    if eid not in index:
        raise ValueError(f'Unknown/unmeasured epistasis evidence {eid}')
    if bool(reference) != bool(test):
        raise ValueError('Both reference and test are required for a directed citation')
    focus = proposed_substitutions(reference, test) if reference and test else ()
    record = index[eid]
    direction = _epistasis_direction(record, focus, eid)
    pa, pb = record['position_a'], record['position_b']
    varying = {POSITIONS.index(pa), POSITIONS.index(pb)}
    fixed = [i for i in range(4) if i not in varying]
    same_background = bool(reference and ''.join(reference[i] for i in fixed) == record['background'])
    in_states = bool(reference and all(reference[POSITIONS.index(pos)] in (sub[0], sub[-1])
                                      for pos, sub in ((pa,record['substitution_a']), (pb,record['substitution_b']))))
    return {'evidence_id':eid, 'reference_variant':reference, 'test_variant':test,
            'target_substitutions':list(focus),
            'historical_substitutions':[record['substitution_a'],record['substitution_b']],
            'background':record['background'], 'same_fixed_background':same_background,
            'anchor_in_record_states':in_states, 'focal_direction':direction,
            'canonical_epsilon':record['epsilon'],
            'epsilon_in_proposed_direction':direction*record['epsilon'],
            **{key:record[key] for key in ('baseline_variant','single_a_variant','single_b_variant',
                                          'double_variant','f00','f10','f01','f11')},
            'scope':'Prior measured corners; not the unmeasured candidate effect'}


def epistasis_reference_options(reference, test, observed, *, index=None, limit=3):
    """Small candidate-specific citation list; no graph or hidden-label access."""
    focus = proposed_substitutions(reference, test)
    if not focus:
        return []
    index = epistasis_evidence_index(observed) if index is None else index
    options = []
    for eid, record in index.items():
        allowed = set()
        for sub in (record['substitution_a'],record['substitution_b']):
            allowed.update((sub,sub[-1]+sub[1:-1]+sub[0]))
        if not all(sub in allowed for sub in focus):
            continue
        options.append(epistasis_reference_facts(eid,reference,test,observed,index=index))
    options.sort(key=lambda r:(-int(r['same_fixed_background']),-int(r['anchor_in_record_states']),
                               -abs(r['epsilon_in_proposed_direction']),r['evidence_id']))
    return options[:limit]


def validate_epistasis_claims(value, observed, *, hypotheses=(), candidates=(), previous_hypotheses=()):
    """Bind cited IDs and numbers to measured corners and the actual target.

    Evidence is rebuilt from visible measurements, never from LLM-returned facts
    or an optional graph snapshot. OFF therefore does not access the KG module.
    """
    by_h = {h['id']: h for h in hypotheses}
    if isinstance(value, dict):
        by_h.update({h['id']: h for h in value.get('hypotheses', [])})
    elif isinstance(value, list):
        by_h.update({h['id']: h for h in value if isinstance(h, dict) and 'matched_comparison' in h})
    by_previous = {h['id']: h for h in previous_hypotheses}
    by_variant = {r['variant']: r for r in candidates}
    index = None

    def lookup(eid, focus, path):
        nonlocal index
        if index is None:
            index = epistasis_evidence_index(observed)
        if eid not in index:
            raise ValueError(f'{path}: unknown/unmeasured epistasis evidence {eid}; four prior measured corners required')
        row = index[eid]
        return row, _epistasis_direction(row, focus, f'{path}: {eid}')

    def walk(item, path='', focus=(), row_ids=()):
        if isinstance(item, dict):
            context = item
            if item.get('hypothesis_id') in by_h:
                context = by_h[item['hypothesis_id']]
            if item.get('previous_id') in by_previous:
                context = by_previous[item['previous_id']]
            if item.get('variant') in by_variant and '.hypothesis_assessments[' not in path:
                context = by_variant[item['variant']]
            pair = context.get('matched_comparison', context)
            ref, test = pair.get('reference_variant'), pair.get('test_variant')
            if ref and test:
                focus = proposed_substitutions(ref, test)
            elif context.get('anchor_variant') and context.get('variant'):
                focus = proposed_substitutions(context['anchor_variant'], context['variant'])
            if 'epistasis_evidence_ids' in item:
                ids = item['epistasis_evidence_ids']
                if not isinstance(ids,list) or any(not isinstance(eid,str) for eid in ids):
                    raise ValueError(f'{path}: epistasis_evidence_ids must be a list of full evidence IDs')
                row_ids = tuple(dict.fromkeys(ids))
                for reference_id in row_ids:
                    lookup(reference_id,focus,path)
            eid = item.get('evidence_id', '')
            if isinstance(eid, str) and eid.startswith('epistasis:'):
                record, direction = lookup(eid, focus, path)
                row_ids = (eid,)
                if 'focal_direction' in item and item['focal_direction'] != direction:
                    raise ValueError(f'{path}: {eid}: focal_direction must be {direction}')
                if 'focal_substitution' in item and item['focal_substitution'] not in focus:
                    raise ValueError(f'{path}: {eid}: focal_substitution differs from the actual target')
                for field in ('epsilon_in_proposed_direction', 'canonical_epsilon', 'epsilon'):
                    if field in item:
                        expected = record['epsilon'] * (direction if field == 'epsilon_in_proposed_direction' else 1)
                        _check_epsilon(item[field], expected, f'{path}.{field}: {eid}')
            for key, child in item.items():
                yield from walk(child, f'{path}.{key}', focus, row_ids)
        elif isinstance(item, list):
            for i, child in enumerate(item):
                yield from walk(child, f'{path}[{i}]', focus, row_ids)
        elif isinstance(item, str):
            yield path, item, focus, row_ids

    for path, text, focus, row_ids in walk(value):
        citations = list(EPI_ID.finditer(text))
        for citation in citations:
            lookup(citation.group(), focus, path)
        for claim in EPSILON.finditer(text):
            # Bind each number within its own sentence, including value-before-ID.
            start = max((m.end() for m in re.finditer(r'[。；;\n]', text[:claim.start()])), default=0)
            end_match = re.search(r'[。；;\n]', text[claim.end():])
            end = claim.end() + end_match.start() if end_match else len(text)
            local = [c for c in citations if c.start() >= start and c.end() <= end]
            preceding = [c for c in local if c.end() <= claim.start()]
            if preceding:
                evidence_id = preceding[-1].group()
            elif len(local) == 1:
                evidence_id = local[0].group()
            else:
                # Hypothetical future controls are not historical epsilon claims.
                if not local and re.search(r'若|如果|假设|预期|阈值', text[start:claim.start()]):
                    continue
                # One explicit ID in the field or this review row is still an
                # unambiguous citation. Punctuation is not an evidence boundary.
                field_ids = set(c.group() for c in citations)
                fallback = field_ids if field_ids else set(row_ids)
                if len(fallback) != 1:
                    raise ValueError(f'{path}: epsilon value requires an unambiguous epistasis evidence ID; '
                                     'supply one epistasis_evidence_ids entry or an inline ID, '
                                     'or use a qualitative explanation without an epsilon value')
                evidence_id = next(iter(fallback))
            record, direction = lookup(evidence_id, focus, path)
            prefix = text[max(start, claim.start() - 12):claim.start()]
            canonical = bool(re.search(r'原始|规范|canonical', prefix, re.IGNORECASE))
            expected = record['epsilon'] * (1 if canonical else direction)
            _check_epsilon(claim.group(1), expected, f'{path}: {evidence_id}')


def sequence_differences(reference, anchor, varying_position=None):
    """Exact, auditable differences; a background omits its varying position."""
    return [{'position': pos, 'reference_residue': a, 'anchor_residue': b}
            for pos, a, b in zip(POSITIONS, reference, anchor)
            if pos != varying_position and a != b]


def validate_background_claims(text, description):
    for pattern_index, pattern in enumerate(BACKGROUND_DIFFERENCES):
        for claim in pattern.finditer(text):
            a, b = claim.group(1, 2)
            fixed = POSITIONS
            if len(a) == 3 or len(b) == 3:
                # A three-residue context is ambiguous without the omitted site.
                # Resolve only from an explicit single-site pair in this text.
                omitted = set()
                for arrow in ARROW.finditer(text):
                    ref, test = arrow.groups()
                    changes = [i for i in range(4) if ref[i] != test[i]]
                    if len(changes) != 1:
                        continue
                    i = changes[0]
                    background = ref[:i] + ref[i + 1:]
                    if background in (a, b):
                        omitted.add(i)
                if len(omitted) != 1:
                    raise ValueError(f'{description}: ambiguous background {a}/{b}; '
                                     'compare full four-site variant IDs explicitly')
                i = omitted.pop()
                fixed = tuple(pos for j, pos in enumerate(POSITIONS) if j != i)
                a = a[:i] + a[i + 1:] if len(a) == 4 else a
                b = b[:i] + b[i + 1:] if len(b) == 4 else b
            actual = {pos for pos, old, new in zip(fixed, a, b) if old != new}
            reported = {int(pos) for pos in re.findall(r'39|40|41|54', claim.group('sites'))}
            exact = pattern_index == 1 or bool(claim.group('only'))
            if not reported <= actual or (exact and reported != actual):
                raise ValueError(f'{description}: {claim.group(1)}/{claim.group(2)} '
                                 f'differ at positions {sorted(actual)}, not {sorted(reported)}; '
                                 'do not attribute a multi-position background change to one site')
    # Comparing two effects with multiple changing background positions does
    # not isolate one of those positions, even when no "only differs" is said.
    arrows = list(ARROW.finditer(text))
    for index in range(1, len(arrows)):
        left, right = arrows[index - 1], arrows[index]
        a, b = left.groups()
        c, d = right.groups()
        changed = [i for i in range(4) if a[i] != b[i]]
        if (len(changed) != 1 or [i for i in range(4) if c[i] != d[i]] != changed
                or substitution(a, b) != substitution(c, d)):
            continue
        differences = sequence_differences(a, c, POSITIONS[changed[0]])
        if len(differences) < 2:
            continue
        stop = arrows[index + 1].start() if index + 1 < len(arrows) else len(text)
        tail = re.split(r'[。；;\n]', text[right.end():stop], maxsplit=1)[0]
        for claim in re.finditer(rf'(?:单独)?(?:归因于|量化|隔离出|分离出)\s*(?P<sites>{SITE_LIST})', tail):
            prefix = re.split(r'[，,]|但|然而', tail[:claim.start()])[-1]
            if re.search(r'(?:不能|不可|无法|不应|不足以)[^，,。；;\n]{0,8}$', prefix):
                continue
            sites = {int(pos) for pos in re.findall(r'39|40|41|54', claim.group('sites'))}
            if len(sites) == 1:
                actual = [item['position'] for item in differences]
                raise ValueError(f'{description}: {a}/{c} differ at background positions {actual}; '
                                 f'comparing {a}->{b} and {c}->{d} cannot isolate position {next(iter(sites))}; '
                                 'require matched-background controls')


def check_direction(tail, actual, description):
    # A limitation AFTER an assertion must not cancel that assertion. Qualifiers
    # apply locally to each effect phrase, not to the whole sentence.
    for clause in re.split(r'[，,。；;\n]|但是|然而|不过|但|而是', tail):
        for effect in EFFECT.finditer(clause):
            prefix = clause[:effect.start()]
            if re.search(r'(?:可能|或许|是否|预期|假设|若|如果|不能证明|无法证明|尚不能证明|'
                         r'未能证明|不足以证明|不能确定|无法确定|不能否定|无法否定|'
                         r'不能排除|无法排除|不一定|未必)'
                         r'[^，,。；;\n]{0,12}$', prefix) and not re.search(
                             r'(?:实测|测得|实际|结果显示)[^，,。；;\n]{0,8}$', prefix):
                continue
            if re.search(r'(?:不能|不可|不得|不应)(?:将|把)?[^，,。；;\n]{0,12}'
                         r'(?:说成|视为|当作|认定为|解读为)[强弱显著明显的]*$', prefix):
                continue
            negated = bool(re.search(r'(?:并非|不是|不具有|没有|并无|无|未见|未显示|未观察到|未|不)'
                                     r'(?:出|为|有|明显|显著|强|弱|的)*$', prefix))
            negated = negated or bool(re.search(r'(?:反驳|推翻|否定|不支持|挑战)[^，,。；;\n]{0,16}$', prefix))
            # Property changes are not a claim about the measured Fitness sign.
            metrics = re.findall(r'fitness|适应度|疏水性|水合指数|hydropathy|电荷|体积|大小|'
                                 r'极性|分子量|uncertainty|不确定性', prefix, re.IGNORECASE)
            if (effect.group() in {'提高', '增加', '提升', '降低', '下降'} and metrics
                    and metrics[-1].lower() not in ('fitness', '适应度')):
                continue
            sign = 1 if effect.group() in POSITIVE_EFFECTS else -1
            contradiction = actual * sign <= 0 if not negated else actual * sign > 0
            if contradiction:
                raise ValueError(f'{description}: claimed effect direction contradicts measured delta {actual:+.6f}')


def substitution(a, b):
    changed = [i for i in range(4) if a[i] != b[i]]
    if len(changed) != 1:
        raise ValueError(f'Cited historical pair is not single-site: {a}->{b}')
    i = changed[0]
    return f'{a[i]}{POSITIONS[i]}{b[i]}'


def text_fields(value, path=''):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from text_fields(item, f'{path}.{key}')
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from text_fields(item, f'{path}[{index}]')


def validate_text_claims(value, observed, *, hypotheses=(), candidates=(), previous_hypotheses=()):
    validate_epistasis_claims(value, observed, hypotheses=hypotheses, candidates=candidates,
                             previous_hypotheses=previous_hypotheses)
    fitness = (observed if isinstance(observed, dict) else
               dict(zip(observed['variant'], observed['fitness'])))
    for path, text in text_fields(value):
        validate_background_claims(text, path)
        if path.endswith('predicted_evidence'):
            continue
        arrows = list(ARROW.finditer(text))
        for index, arrow in enumerate(arrows):
            a, b = arrow.groups()
            if a not in fitness or b not in fitness:
                continue  # A proposed/unmeasured pair is not historical evidence.
            stop = arrows[index + 1].start() if index + 1 < len(arrows) else len(text)
            tail = re.split(r'[。；;\n]', text[arrow.end():stop], maxsplit=1)[0]
            actual = float(fitness[b]) - float(fitness[a])
            prefix = pair_claim_prefix(text, arrows, index)
            if prefix and not PREDICTION_PREFIX.search(prefix):
                check_direction(prefix, actual, f'{path}: {a}->{b}')
            for offset, claim in measured_pair_clauses(tail, prefix):
                check_direction((prefix if offset == 0 else '') + claim, actual, f'{path}: {a}->{b}')
                for match in DELTA.finditer(claim):
                    # A reported range is not an equality to its endpoint.
                    if re.match(r'\s*[-–~～至]', claim[match.end():]):
                        continue
                    number = match.group(1).replace('−', '-')
                    tolerance = 0.5 * 10 ** Decimal(number).as_tuple().exponent + 1e-5
                    if abs(float(number) - actual) > tolerance:
                        raise ValueError(
                            f'{path}: {a}->{b} ({substitution(a, b)}) has '
                            f'delta_fitness={actual:+.6f}, not {number}; use test minus reference')
        for match in CONTEXT.finditer(text):
            background, old, position, new = match.groups()
            index = POSITIONS.index(int(position))
            a = background[:index] + old + background[index:]
            b = background[:index] + new + background[index:]
            if a in fitness and b in fitness:
                tail = re.split(r'[。；;\n]', text[match.end():], maxsplit=1)[0]
                for _, claim in measured_pair_clauses(tail):
                    check_direction(claim, float(fitness[b]) - float(fitness[a]), f'{path}: {a}->{b}')


def validate_hypothesis_claims(hypotheses, observed):
    validate_text_claims(hypotheses, observed)
    fitness = (observed if isinstance(observed, dict) else
               dict(zip(observed['variant'], observed['fitness'])))
    for hypothesis in hypotheses:
        pair = hypothesis['matched_comparison']
        expected = substitution(pair['reference_variant'], pair['test_variant'])
        for field in ('knowledge_rationale', 'observed_evidence'):
            for arrow in ARROW.finditer(hypothesis.get(field, '')):
                a, b = arrow.groups()
                if a in fitness and b in fitness and substitution(a, b) != expected:
                    raise ValueError(
                        f"{hypothesis['id']}.{field}: {a}->{b} is {substitution(a, b)}, "
                        f'not evidence for the same substitution {expected}; '
                        'move different-substitution comparisons to design_guidance and label them explicitly')
