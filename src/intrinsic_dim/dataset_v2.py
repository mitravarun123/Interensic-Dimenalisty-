"""Version-2 task-family design. No lexical paraphrase is a new family."""
from pathlib import Path
from fractions import Fraction
from itertools import permutations
import json
import re
import numpy as np
from .dataset import CATEGORIES,NAMES,validate

DESIGN_VERSION='2.0'
SUFFIX=' Give only the final answer, without an explanation.'
ARITHMETIC=['addition','subtraction','multiplication','exact_division','remainder','power','linear_evaluation','fraction_addition','percentage','weighted_sum','multiply_subtract','difference_of_squares','nested_parentheses','arithmetic_series','mean','numerical_comparison','ratio_reduction','linear_equation','square_root','unit_conversion']
REASONING=['strict_order','four_item_rank','class_inclusion','contraposition','forward_implication','seat_constraints','conjunction','disjunction','exactly_one','threshold','chained_rules','nested_boolean','biconditional','negation_chain','graph_reachability','truth_and_lies','set_difference','dependency_schedule','necessary_condition','color_constraints']
CODING=['sum','max','sorted','reverse_sort','reverse_slice','stride_slice','dict_get','list_index','list_count','membership','modulo','filter_comprehension','map_comprehension','nested_comprehension','zip','enumerate','string_replace','split','all','any']

def arithmetic(f,rng):
    a,b,c=rng.integers(2,15,size=3).tolist(); n=int(rng.integers(3,12)); label=ARITHMETIC[f]
    prompts=[
      f'Two collections contain {a} and {b} objects respectively. Calculate the total number of objects obtained by adding those quantities.',
      f'A collection starts with {a+b} objects, and {b} objects are removed. Calculate the number of objects remaining after this subtraction.',
      f'There are {a} equal groups, each containing {b} objects. Calculate the total number of objects using multiplication.',
      f'Divide {a*b} objects equally into {b} groups without any remainder. Calculate how many objects belong to each group.',
      f'When the integer {a*b+1} is divided by {b}, calculate the remainder left over after taking all complete groups.',
      f'Evaluate the integer power {a} raised to {c}. Return the exact integer value of this arithmetic expression.',
      f'For the expression {a} times x plus {b}, substitute x equal to {c}. Calculate the resulting value of the expression.',
      f'Add the exact fractions {a}/{b} and {c}/{b}. Express the sum as a reduced fraction, or an integer if the denominator reduces to one.',
      f'Calculate {a} percent of {b*100}. Interpret percent in the usual arithmetic sense and return the exact numerical value.',
      f'Calculate the weighted sum formed from {a} copies of {b} and two copies of {c}. What is the resulting total?',
      f'First multiply {a} by {b}, then subtract {c} from the product. Calculate the final integer result of these two operations.',
      f'Calculate the square of {a}, then subtract the square of {b}. What is the resulting difference of squares?',
      f'Evaluate the quantity ({a} plus {b}) multiplied by ({c} plus 2). Perform the additions before the multiplication.',
      f'Add all consecutive positive integers from 1 through {n}, including both endpoints. Calculate the exact sum of this arithmetic series.',
      f'Three measurements are {a}, {a+3}, and {a+6}. Calculate their arithmetic mean by dividing their sum by three.',
      f'Compare the two exact numerical quantities {a*b} and {a*c}. Calculate the larger of these quantities and return its value.',
      f'Reduce the numerical ratio {a*b} to {a*c} to a fraction in lowest terms. Return a reduced fraction, or an integer if appropriate.',
      f'Solve the equation {a} times x plus {b} equals {a*c+b}. Calculate the numerical value of x that satisfies this equation.',
      f'Calculate the principal nonnegative square root of {a*a}. Return the exact integer whose square equals the given value.',
      f'Convert {a} metres into centimetres using the relation one metre equals 100 centimetres. Calculate the numerical count of centimetres.'
    ]
    answers=[a+b,a,a*b,a,1,a**c,a*c+b,Fraction(a+c,b),a*b,a*b+2*c,a*b-c,a*a-b*b,(a+b)*(c+2),n*(n+1)//2,a+3,max(a*b,a*c),Fraction(b,c),c,a,a*100]
    return label,prompts[f],str(answers[f]),[str(a),str(b),str(c)]

def reasoning(f,rng,split):
    pool=NAMES[8:] if split=='evaluation' else NAMES[:8]
    a,b,c,d=rng.choice(pool,4,replace=False).tolist(); p,q,r=rng.integers(0,2,size=3).astype(bool).tolist()
    t=lambda v:'true' if v else 'false'; yes=lambda v:'yes' if v else 'no'
    # Each truth answer follows the explicit rule, not a generated language-model answer.
    prompts=[
      f'{a} is older than {b}, and {b} is older than {c}. Assuming these are strict age comparisons, who is the oldest of the three people?',
      f'{a} stands before {b}, who stands before {c}, who stands before {d}. These four people form a line. Who is second?',
      f'All members of class {a} belong to class {b}, and all members of class {b} belong to class {c}. Must every member of {a} also belong to {c}?',
      f'If {a} enters a room, its alarm is active. The alarm is not active. Under this implication, is it possible that {a} entered the room?',
      f'Whenever {a} registers, {b} receives a notification. It is a fact that {a} registered. Must {b} receive a notification under the stated rule?',
      f'{a}, {b}, and {c} occupy three seats. {a} is not middle. {b} is left of {c}; {c} is right of {a}. Who is middle?',
      f'The gate opens exactly when all three inputs are true. The inputs are {t(p)}, {t(q)}, and {t(r)}. Does the gate open under this rule?',
      f'The lamp turns on exactly when at least one input is true. The inputs are {t(p)}, {t(q)}, and {t(r)}. Does the lamp turn on?',
      f'The switch activates exactly when one input is true and the other two are false. The inputs are {t(p)}, {t(q)}, and {t(r)}. Does it activate?',
      f'An alert appears exactly when at least two of its three tests are true. The test values are {t(p)}, {t(q)}, and {t(r)}. Does the alert appear?',
      f'If {a} arrives then {b} arrives; if {b} arrives and {c} arrives then {d} arrives. Both {a} and {c} arrive. Must {d} arrive under these rules?',
      f'An output is true exactly when (P or Q) and not R is true. P is {t(p)}, Q is {t(q)}, and R is {t(r)}. Is the output true?',
      f'Signal {a} is true if and only if signal {b} is true. Signal {a} is {t(p)}. Is signal {b} true under this biconditional rule?',
      f'Boolean signal {a} starts as {t(p)}. Three logical negations are applied in succession, each reversing its truth value. Is the resulting signal true?',
      f'A directed graph contains edges {a} to {b} and {b} to {c}, with no other edges. Is there a directed path from {a} to {c}?',
      f'One of {a} and {b} always tells the truth, and the other always lies. {a} says that both of them are liars. Which person always tells the truth?',
      f'A set contains {a}, {b}, and {c}. Remove every member of the subset containing {b} and {c}. Which named member remains in the resulting set?',
      f'Task {a} must finish before {b} begins, and {b} must finish before {c} begins. Each task is performed once. Which task must finish first?',
      f'Access is allowed only if credential {a} is valid. That credential is valid, but no sufficient condition for access is given. Is access necessarily allowed?',
      f'{a}, {b}, and {c} have different colors: red, green, or blue. {a} is red and {b} is not blue. Who is blue?'
    ]
    answers=[a,b,'yes','no','yes',b,yes(p and q and r),yes(p or q or r),yes(sum([p,q,r])==1),yes(sum([p,q,r])>=2),'yes',yes((p or q) and not r),yes(p),yes(not p),'yes',b,a,a,'no',c]
    # Independently enumerate the two constraint families.
    seat=[order for order in permutations([a,b,c]) if order[1]!=a and order.index(b)<order.index(c) and order.index(a)<order.index(c)]
    color=[dict(zip([a,b,c],order)) for order in permutations(['red','green','blue']) if order[0]=='red' and order[1]!='blue' and order[2]!='green']
    assert len(seat)==1 and seat[0][1]==b and len(color)==1 and color[0][c]=='blue'
    return REASONING[f],prompts[f],answers[f],[a,b,c,d]

def coding(f,rng,split):
    v=rng.choice(np.arange(1,10),4,replace=False).tolist(); a,b,c,d=v
    alphabet=list('mnopqrs' if split=='evaluation' else 'abcdefg'); word=''.join(rng.choice(alphabet,6).tolist())
    dictionary={'left':a,'right':b}; pairs=list(zip(v,[2,4,6,8])); enumerated=list(enumerate(v,start=1))
    expressions=[f'sum({v})',f'max({v})',f'sorted({v})',f'sorted({v}, reverse=True)',f'{v}[::-1]',f'{v}[::2]',f'{dictionary}.get("left")',f'{v}.index({c})',f'{v+[a]}.count({a})',f'{c} in {v}',f'{a*b+1} % {b}',f'[x for x in {v} if x > 4]',f'[x * 2 for x in {v}]',f'[x + y for x in {v[:2]} for y in [1, 2]]',f'list(zip({v}, [2, 4, 6, 8]))',f'list(enumerate({v}, start=1))',f'{word!r}.replace({word[0]!r}, "z")',f'{(word[:3]+","+word[3:])!r}.split(",")',f'all(x > 4 for x in {v})',f'any(x > 4 for x in {v})']
    answers=[sum(v),max(v),sorted(v),sorted(v,reverse=True),v[::-1],v[::2],dictionary.get('left'),v.index(c),(v+[a]).count(a),c in v,(a*b+1)%b,[x for x in v if x>4],[x*2 for x in v],[x+y for x in v[:2] for y in [1,2]],pairs,enumerated,word.replace(word[0],'z'),(word[:3]+','+word[3:]).split(','),all(x>4 for x in v),any(x>4 for x in v)]
    prompt=f'In Python 3, evaluate {expressions[f]}. Return the resulting value as a Python literal.'
    return CODING[f],prompt,repr(answers[f]),[word]

def generate_v2(count=640,seed=42):
    if count%80 or not 160<=count<=640: raise ValueError('Version 2 supports balanced multiples of 80 from 160 to 640; extend fixed factual fixtures for larger studies')
    fixtures=json.loads((Path(__file__).resolve().parents[2]/'data/factual_fixtures_v2.json').read_text())
    rng=np.random.default_rng(seed); per_family=count//80; rows=[]; seen=set()
    for category in CATEGORIES:
        for f in range(20):
            split='evaluation' if f%2 else 'exploratory'
            for item in range(per_family):
                for attempt in range(1000):
                    source=None
                    if category=='facts':
                        fixture=fixtures[f];subject,answer=fixture['items'][item];family=fixture['family'];prompt=fixture['question'].format(subject=subject);entities=[subject];source=fixture['source']
                    elif category=='arithmetic':family,prompt,answer,entities=arithmetic(f,rng)
                    elif category=='reasoning':family,prompt,answer,entities=reasoning(f,rng,split)
                    else:family,prompt,answer,entities=coding(f,rng,split)
                    if category=='facts': prompt += ' Preserve the official notation in your response.'
                    if category=='arithmetic': prompt += ' Use exact arithmetic and simplify the result.'
                    text=prompt+SUFFIX
                    if text not in seen:break
                else: raise ValueError(f'Family {category}/{f} exhausted; no duplicate prompts are permitted')
                seen.add(text)
                entities=[e for e in entities if re.search(r'\b'+re.escape(e)+r'\b',prompt)]
                rows.append({'prompt_id':f'v2_{category}_{family}_{item:03d}','task_category':category,'template_id':f'v2_{category}_{family}','template_family_index':f,'prompt':text,'expected_answer':answer,'token_count':None,'token_count_method':'unavailable_until_tokenized','split':split,'generation_seed':seed,'entities':entities,'answer_format':'concise_answer','answer_type':'python_literal' if category=='coding' else 'short_text_or_number','answer_verification':'fixed_authoritative_fixture' if category=='facts' else 'computed_rules_or_python_semantics','ground_truth_source':source,'semantic_item_id':f"{category}:{family}:{entities[0] if entities else 'rule'}:{item}",'design_version':DESIGN_VERSION,'review_status':'programmatically_checked_human_review_pending'})
    return validate(rows)
