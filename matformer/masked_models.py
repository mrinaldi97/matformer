"""
matformer/masked_models.py
"""   
import random
def maskerator(input_ids,MASK_TOKEN=0,substitution_rate=0.3):
        # The input_ids should already start with the CLS token, expected at position #0
        output=[input_ids[0]]
        cloze_mask=[0]
        for token in input_ids[1:]:
            destino=1-random.random()
            if destino<=substitution_rate:
                if destino<=1:
                    output.append(MASK_TOKEN)
                    cloze_mask.append(token)
                else:
                    if destino<=0.5:
                        #Sostituisco con Random (non implementato)
                        output.append(random.randint(0,vocabulary_length-1))
                        cloze_mask.append(token)
                    else:
                        #Non sostituisco (non implementato)
                        output.append(token)
                        cloze_mask.append(token)                                
            else:
                #Il token non viene sostituito, Albertone dovrà prevedere ZERO
                output.append(token)
                cloze_mask.append(0)
        assert len(output)==len(cloze_mask)
        return output,cloze_mask
           
