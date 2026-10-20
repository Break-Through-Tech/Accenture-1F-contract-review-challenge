import json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRAIN = Path('train_separate_questions.json')
TEST = Path('test.json')
OUT = Path('week3_outputs')
(OUT/'results').mkdir(parents=True, exist_ok=True)
(OUT/'data').mkdir(parents=True, exist_ok=True)
WORD_RE = re.compile(r'\S+')

def count_words(text):
    return len(WORD_RE.findall(text))

def word_spans(text):
    return [(m.start(), m.end()) for m in WORD_RE.finditer(text)]

def fixed_word_chunks(text, max_words=500, overlap_words=0):
    spans = word_spans(text)
    if not spans:
        return []
    step = max_words - overlap_words
    if step <= 0:
        raise ValueError('overlap_words must be smaller than max_words')
    chunks, i, cid = [], 0, 0
    while i < len(spans):
        j = min(i + max_words, len(spans))
        s, e = spans[i][0], spans[j-1][1]
        chunks.append({'chunk_id':cid,'start_char':s,'end_char':e,'word_count':j-i,'text':text[s:e]})
        cid += 1
        if j == len(spans):
            break
        i += step
    return chunks

def paragraph_units(text):
    units=[]
    for m in re.finditer(r'\S(?:.*?\S)?(?=\n\s*\n|\Z)', text, flags=re.S):
        if m.group(0).strip():
            units.append((m.start(), m.end()))
    if not units and text.strip():
        s = len(text)-len(text.lstrip())
        e = len(text.rstrip())
        units=[(s,e)]
    return units

def split_large_unit(text, start, end, max_words):
    segment=text[start:end]
    cuts=[0]
    for m in re.finditer(r'(?<=[.!?;])\s+(?=[A-Z0-9(\[])', segment):
        cuts.append(m.end())
    cuts.append(len(segment))
    sentences=[]
    for a,b in zip(cuts[:-1],cuts[1:]):
        s,e=start+a,start+b
        if text[s:e].strip():
            sentences.append((s,e))
    pieces=[]; current=[]; current_words=0
    for s,e in sentences:
        sw=count_words(text[s:e])
        if sw > max_words:
            if current:
                pieces.append((current[0][0],current[-1][1]))
                current=[]; current_words=0
            for ch in fixed_word_chunks(text[s:e],max_words,0):
                pieces.append((s+ch['start_char'],s+ch['end_char']))
        elif current_words+sw <= max_words:
            current.append((s,e)); current_words += sw
        else:
            pieces.append((current[0][0],current[-1][1]))
            current=[(s,e)]; current_words=sw
    if current:
        pieces.append((current[0][0],current[-1][1]))
    return pieces

def paragraph_aware_chunks(text, max_words=350, overlap_words=50):
    units=[]
    for s,e in paragraph_units(text):
        if count_words(text[s:e]) <= max_words:
            units.append((s,e))
        else:
            units.extend(split_large_unit(text,s,e,max_words))
    packed=[]; current=[]; current_words=0
    for s,e in units:
        uw=count_words(text[s:e])
        if not current or current_words+uw <= max_words:
            current.append((s,e)); current_words += uw
        else:
            packed.append((current[0][0],current[-1][1]))
            current=[(s,e)]; current_words=uw
    if current:
        packed.append((current[0][0],current[-1][1]))
    words=word_spans(text)
    starts=np.array([s for s,_ in words],dtype=int) if words else np.array([],dtype=int)
    chunks=[]
    for cid,(s,e) in enumerate(packed):
        s2=s
        if cid>0 and overlap_words>0 and len(words):
            pos=int(np.searchsorted(starts,s,side='left'))
            pos=max(0,pos-overlap_words)
            s2=words[pos][0]
        chunks.append({'chunk_id':cid,'start_char':int(s2),'end_char':int(e),'word_count':count_words(text[s2:e]),'text':text[s2:e]})
    return chunks

STRATEGIES={
    'fixed_300_no_overlap':lambda t:fixed_word_chunks(t,300,0),
    'fixed_500_no_overlap':lambda t:fixed_word_chunks(t,500,0),
    'fixed_350_overlap_50':lambda t:fixed_word_chunks(t,350,50),
    'fixed_500_overlap_100':lambda t:fixed_word_chunks(t,500,100),
    'paragraph_300_overlap_50':lambda t:paragraph_aware_chunks(t,300,50),
    'paragraph_325_overlap_50':lambda t:paragraph_aware_chunks(t,325,50),
    'paragraph_350_overlap_50':lambda t:paragraph_aware_chunks(t,350,50),
    'paragraph_400_overlap_75':lambda t:paragraph_aware_chunks(t,400,75),
    'paragraph_500_no_overlap':lambda t:paragraph_aware_chunks(t,500,0),
    'paragraph_500_overlap_75':lambda t:paragraph_aware_chunks(t,500,75),
    'paragraph_600_overlap_100':lambda t:paragraph_aware_chunks(t,600,100),
}

def extract_contracts(dataset, split):
    out=[]
    for item in dataset['data']:
        for pidx,p in enumerate(item['paragraphs']):
            answers=[]
            for qa in p['qas']:
                for a in qa.get('answers',[]):
                    answers.append({'start':a['answer_start'],'end':a['answer_start']+len(a['text']),'text':a['text'],'qa_id':qa['id']})
            out.append({'split':split,'title':item['title'],'paragraph_index':pidx,'context':p['context'],'answers':answers})
    return out

def evaluate_strategy(contracts,name,fn):
    details=[]; chunk_words=[]; total_words=total_chunk_words=total_spans=covered=0
    for c in contracts:
        chunks=fn(c['context']); cw=count_words(c['context'])
        total_words += cw
        total_chunk_words += sum(ch['word_count'] for ch in chunks)
        chunk_words.extend(ch['word_count'] for ch in chunks)
        ccov=0
        for a in c['answers']:
            total_spans += 1
            ok=any(ch['start_char']<=a['start'] and ch['end_char']>=a['end'] for ch in chunks)
            if ok:
                covered += 1; ccov += 1
        details.append({'split':c['split'],'strategy':name,'contract_title':c['title'],'contract_words':cw,'chunk_count':len(chunks),'span_count':len(c['answers']),'covered_span_count':ccov,'span_coverage':ccov/len(c['answers']) if c['answers'] else np.nan})
    total_chunks=sum(d['chunk_count'] for d in details)
    return {
        'strategy':name,'contracts':len(contracts),'total_chunks':total_chunks,
        'avg_chunks_per_contract':total_chunks/len(contracts),
        'avg_chunk_words':float(np.mean(chunk_words)),
        'median_chunk_words':float(np.median(chunk_words)),
        'p95_chunk_words':float(np.percentile(chunk_words,95)),
        'max_chunk_words':int(max(chunk_words)),
        'span_coverage':covered/total_spans,'covered_spans':covered,'total_spans':total_spans,
        'split_spans':total_spans-covered,'word_redundancy_ratio':total_chunk_words/total_words,
    }, pd.DataFrame(details)

def get_category(qa):
    m=re.search(r'related to "([^"]+)"',qa.get('question',''))
    if m: return m.group(1)
    return re.sub(r'_\d+$','',qa['id'].split('__',1)[1])

def build_chunk_dataset(dataset, split, chunk_fn):
    rows=[]
    for item in dataset['data']:
        for pidx,p in enumerate(item['paragraphs']):
            text=p['context']; anns=[]
            for qa in p['qas']:
                cat=get_category(qa)
                for a in qa.get('answers',[]):
                    anns.append({'category':cat,'start':a['answer_start'],'end':a['answer_start']+len(a['text']),'text':a['text']})
            for ch in chunk_fn(text):
                full=[a for a in anns if ch['start_char']<=a['start'] and ch['end_char']>=a['end']]
                overlap=[a for a in anns if a['start']<ch['end_char'] and a['end']>ch['start_char']]
                rows.append({'split':split,'contract_title':item['title'],'paragraph_index':pidx,'chunk_id':ch['chunk_id'],'start_char':ch['start_char'],'end_char':ch['end_char'],'word_count':ch['word_count'],'chunk_text':ch['text'],'fully_contained_categories':' | '.join(sorted(set(a['category'] for a in full))),'fully_contained_span_count':len(full),'overlapping_categories':' | '.join(sorted(set(a['category'] for a in overlap))),'overlapping_span_count':len(overlap)})
    return pd.DataFrame(rows)

def main():
    train=json.loads(TRAIN.read_text(encoding='utf-8'))
    test=json.loads(TEST.read_text(encoding='utf-8'))
    train_contracts=extract_contracts(train,'train')
    test_contracts=extract_contracts(test,'test')
    summaries=[]; details=[]
    for split_name,contracts in [('train',train_contracts),('test',test_contracts)]:
        for name,fn in STRATEGIES.items():
            s,d=evaluate_strategy(contracts,name,fn)
            s['split']=split_name
            summaries.append(s); details.append(d)
    summary_df=pd.DataFrame(summaries)
    detail_df=pd.concat(details,ignore_index=True)
    summary_df.to_csv(OUT/'results'/'chunking_strategy_comparison.csv',index=False)
    detail_df.to_csv(OUT/'results'/'chunking_contract_detail.csv',index=False)
    rec=STRATEGIES['paragraph_350_overlap_50']
    build_chunk_dataset(train,'train',rec).to_csv(OUT/'data'/'train_chunks_paragraph350_overlap50.csv',index=False)
    build_chunk_dataset(test,'test',rec).to_csv(OUT/'data'/'test_chunks_paragraph350_overlap50.csv',index=False)
    print(summary_df[['split','strategy','avg_chunks_per_contract','avg_chunk_words','span_coverage','split_spans','word_redundancy_ratio']].sort_values(['split','span_coverage'],ascending=[True,False]).to_string(index=False))

if __name__=='__main__':
    main()
