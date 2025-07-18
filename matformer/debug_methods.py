from matformer.tokenizers import MatformerTokenizer

def train_debug_print(_input, output, model_cfg, tokenizer, varlen_strategy):
    """
    Una funzione rozza di debug per stampare a video ciò che entra nel modello e l'obiettivo di apprendimento
    """
    superDebug=False
    if superDebug:
        print("--- SUPER DEBUG ---")
        print(f"Il tokenizzatore è: {tokenizer}")
        print(f"Sta entrando un {type(_input)}")
        print(f"L'obiettivo è un {type(output)}")    
    assert isinstance(tokenizer,MatformerTokenizer)
    _input=_input.tolist()
    output=output.tolist()
    if superDebug:
        print(f"Dimensione lista input: {len(_input)}")
        #print(f"Primo elemento lista input: {_input[0]}")
    with open("DEBUG_input.txt",'a') as i, open("DEBUG_output.txt",'a') as o:
        if superDebug: print("\nDecodifico l'input...")
        _input=tokenizer.decode(_input)
        if superDebug: print(_input)
        i.write(_input+'\n')
        if superDebug: print("\n Decodifico l'output...")
        for contatore,obiettivo_masked in enumerate(output):
                o.write(f"{contatore}: {tokenizer.decode(obiettivo_masked)}\n")
