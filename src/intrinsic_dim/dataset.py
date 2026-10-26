"""Explicit, reproducible template generation with computed reference answers."""
from collections import Counter
from pathlib import Path
import json
import re
import numpy as np
from .io import write_json

CATEGORIES=['facts','arithmetic','reasoning','coding']
REQUIRED=['prompt_id','task_category','template_id','prompt','expected_answer','token_count','split','generation_seed','entities','answer_format','token_count_method']
NAMES=['Alice','Bob','Charlie','Diana','Evan','Fatima','Grace','Hugo','Iris','Jules','Kiran','Lina','Maya','Noah','Omar','Priya']
# Atomic-number/symbol fixtures cross-checked with IUPAC periodic table; no generated truth.
ELEMENTS=[('hydrogen','H'),('helium','He'),('lithium','Li'),('beryllium','Be'),('boron','B'),('carbon','C'),('nitrogen','N'),('oxygen','O'),('fluorine','F'),('neon','Ne'),('sodium','Na'),('magnesium','Mg'),('aluminium','Al'),('silicon','Si'),('phosphorus','P'),('sulfur','S'),('chlorine','Cl'),('argon','Ar'),('potassium','K'),('calcium','Ca'),('scandium','Sc'),('titanium','Ti'),('vanadium','V'),('chromium','Cr'),('manganese','Mn'),('iron','Fe'),('cobalt','Co'),('nickel','Ni'),('copper','Cu'),('zinc','Zn'),('gallium','Ga'),('germanium','Ge'),('arsenic','As'),('selenium','Se'),('bromine','Br'),('krypton','Kr'),('rubidium','Rb'),('strontium','Sr'),('yttrium','Y'),('zirconium','Zr'),('niobium','Nb'),('molybdenum','Mo'),('technetium','Tc'),('ruthenium','Ru'),('rhodium','Rh'),('palladium','Pd'),('silver','Ag'),('cadmium','Cd'),('indium','In'),('tin','Sn')]

def template(category, family, i, rng):
    a,b,c=rng.choice(NAMES,3,replace=False).tolist(); u,v=rng.integers(2,100,size=2).tolist()
    if category=='facts':
        j=i%len(ELEMENTS); name,symbol=ELEMENTS[j]; number=j+1
        texts=[f'What is the chemical symbol for {name}?',f'Which element has atomic number {number}?',f'What is the atomic number of {name}?',f'Name the element represented by {symbol}.',f'Recall the symbol used for the element {name}.',f'Identify the element at position {number} in the periodic table.',f'State how many protons a neutral {name} atom has.',f'Which element is denoted by the chemical symbol {symbol}?']
        answers=[symbol,name,str(number),name,symbol,name,str(number),name]
        return texts[family],answers[family],[name], 'iupac_fixture'
    if category=='arithmetic':
        texts=[f'What is {u} + {v}?',f'Calculate {u} multiplied by {v}.',f'What is {u*v} divided by {v}?',f'If x = {u}, what is {v}x + 5?',f'Subtract {v} from {u+v}.',f'Compute ({u} + {v}) times 2.',f'What is the remainder when {u*v+1} is divided by {v}?',f'Square {u} and subtract {v}.']
        answers=[u+v,u*v,u,v*u+5,u,2*(u+v),1,u*u-v]
        return texts[family],str(answers[family]),[str(u),str(v)],'computed_integer'
    if category=='reasoning':
        flags=rng.integers(0,2,size=2).astype(bool); p,q=flags.tolist()
        state=lambda f:'true' if f else 'false'
        texts=[f'{a} is older than {b}. {b} is older than {c}. Who is oldest?',f'All {a} objects are {b} objects. All {b} objects are {c} objects. Are all {a} objects {c} objects?',f'An alarm activates exactly when both sensors trigger. Sensor {a} is {state(p)} and sensor {b} is {state(q)}. Does it activate?',f'{a} is left of {b}, and {b} is left of {c}. Who is furthest right?',f'The lamp is on exactly when at least one switch is on. Switch {a} is {state(p)} and switch {b} is {state(q)}. Is the lamp on?',f'{a} has more coins than {b}. {c} has fewer coins than {b}. Who has the fewest coins?',f'A gate opens exactly when one of two inputs is true, but not both. Input {a} is {state(p)} and input {b} is {state(q)}. Does it open?',f'{a} finished before {b}; {b} finished before {c}. Who finished second?']
        answers=[a,'yes','yes' if p and q else 'no',c,'yes' if p or q else 'no',c,'yes' if p!=q else 'no',b]
        return texts[family],answers[family],[a,b,c], 'verified_relational_or_boolean'
    values=rng.integers(-20,50,size=5).tolist(); word=''.join(rng.choice(list('abcdefghijk'),6))
    texts=[f'In Python, what is sum({values})?',f'What does Python max({values}) return?',f'What is the output of {word!r}[::-1] in Python?',f'What does sorted({values}) return in Python?',f'What is len({word!r}) in Python?',f'What is the output of [x * 2 for x in {values}]?',f'What does min({values}) return in Python?',f'What does list(reversed({values})) return in Python?']
    answers=[str(sum(values)),str(max(values)),word[::-1],str(sorted(values)),str(len(word)),str([2*x for x in values]),str(min(values)),str(values[::-1])]
    return texts[family],answers[family],[word], 'computed_python_semantics'

def generate(count=200, seed=42):
    if count<4 or count%4: raise ValueError('target_count must be a positive multiple of four >=4')
    rng=np.random.default_rng(seed); rows=[]; seen=set()
    for category in CATEGORIES:
        for i in range(count//4):
            family=i%8
            for attempt in range(10000):
                # Cycling factual bank with changing family extends to 400 unique facts.
                prompt,answer,entities,verification=template(category,family,i+attempt*7,rng)
                text=prompt+' Reply with only the answer.'
                if text not in seen: break
            else: raise ValueError('Template bank exhausted; expand facts before increasing count')
            seen.add(text)
            rows.append({'prompt_id':f'{category}_{i:05d}','task_category':category,'template_id':f'{category}_family_{family}','template_family_index':family,'prompt':text,'expected_answer':answer,'token_count':None,'token_count_method':'unavailable_until_tokenized','split':'evaluation' if family>=6 else 'exploratory','generation_seed':seed,'entities':entities,'answer_format':'short_answer','answer_verification':verification})
    validate(rows)
    return rows

def save_dataset(path, rows):
    validate(rows); path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))

def load_dataset(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    validate(rows); return rows

def validate(rows):
    if not rows: raise ValueError('Dataset is empty')
    ids=set(); texts=set(); families={}
    for r in rows:
        missing=set(REQUIRED)-r.keys()
        if missing: raise ValueError(f'Missing fields: {sorted(missing)}')
        if r['task_category'] not in CATEGORIES or r['split'] not in ['evaluation','exploratory']: raise ValueError('Invalid task or split')
        if not isinstance(r['prompt'],str) or not r['prompt'].strip() or not r['prompt_id'] or not r['template_id']: raise ValueError('Empty prompt or identifier')
        if r['token_count'] is not None and (not isinstance(r['token_count'],int) or r['token_count']<1): raise ValueError('Invalid token count')
        key=' '.join(r['prompt'].lower().split())
        if r['prompt_id'] in ids or key in texts: raise ValueError('Duplicate ID or normalized prompt')
        ids.add(r['prompt_id']); texts.add(key)
        family=(r['task_category'],r['template_id'])
        if family in families and families[family]!=r['split']: raise ValueError('Template family leaks across split')
        families[family]=r['split']
    return rows

def audit(rows):
    # Token Jaccard is a diagnostic heuristic, not a guarantee of semantic independence.
    bags=[set(re.findall(r'\w+',r['prompt'].lower())) for r in rows]; near=[]
    for i in range(len(rows)):
        for j in range(i):
            union=bags[i]|bags[j]
            similarity=len(bags[i]&bags[j])/len(union) if union else 1
            if similarity>=.85: near.append([rows[j]['prompt_id'],rows[i]['prompt_id'],round(similarity,3)])
    groups={}
    for category in CATEGORIES:
        subset=[r for r in rows if r['task_category']==category]; lengths=[r['token_count'] for r in subset if r['token_count'] is not None]
        groups[category]={'count':len(subset),'families':len(set(r['template_id'] for r in subset)),'family_counts':dict(Counter(r['template_id'] for r in subset)),'families_per_split':{split:len({r['template_id'] for r in subset if r['split']==split}) for split in ['exploratory','evaluation']},'splits':dict(Counter(r['split'] for r in subset)),'token_count_available':len(lengths),'token_length_quantiles':np.quantile(lengths,[0,.25,.5,.75,1]).tolist() if lengths else None,'character_length_quantiles':np.quantile([len(r['prompt']) for r in subset],[0,.5,1]).tolist() if subset else None,'answer_formats':dict(Counter(r['answer_format'] for r in subset))}
    warnings=['Equal counts do not ensure matching.','Template-family bootstrap is conditional; independence of semantic families is an assumption, not established by unique IDs.','Shared vocabulary, answer type and domain remain potential confounds.']
    if not all(r.get('design_version')=='2.0' for r in rows): warnings += ['Legacy pilot: chemical recall and code prediction; too few held-out families.']
    duplicates=len(rows)-len(set(' '.join(r['prompt'].lower().split()) for r in rows))
    return {'categories':groups,'exact_normalized_duplicates':duplicates,'near_duplicate_pairs':near,'near_duplicate_threshold':.85,'missing_fields':[{'prompt_id':r.get('prompt_id'),'fields':sorted(set(REQUIRED)-r.keys())} for r in rows if set(REQUIRED)-r.keys()],'repeated_entities':dict(Counter(e for r in rows for e in r.get('entities',[]))),'token_methods':dict(Counter(r['token_count_method'] for r in rows)),'warnings':warnings}
