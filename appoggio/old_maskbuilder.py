class MaskBuilder():
    """
    Handles block mask creation logic for FlexAttention implementation
    It should be moved to 'transformer_functions.py'
    """
    def __init__(self, config):
        self.config = config
    def _mask_fn_causal(self, b, h, q_idx, kv_idx, **kwargs):
        return q_idx >= kv_idx
    def _mask_fn_sliding(self, b, h, q_idx, kv_idx,window_size, **kwargs):
        return kv_idx > q_idx - window_size
    def _mask_fn_document(self, b, h, q_idx, kv_idx, document_mask, **kwargs):
        if document_mask is None:
            ##print("WARNING: Document block-mask function called without document_mask data.")
            return True # Permissive
        return document_mask[b][q_idx] == document_mask[b][kv_idx]
    def _maskfn_cloze(self, b, h, q_idx, kv_idx, cloze_mask, **kwargs): 
        if cloze_mask is None:
            ##print("WARNING: Cloze block-mask function called without cloze_mask.") 
            return True # Permissive
        return cloze_mask[b][q_idx]
    def build_mask_tensor(self, attention_types, query=None, kv=None, batch_size=None, num_heads=None,                             
                             is_sliding=False,
                             document_mask=None,
                             cloze_mask=None, nested=True, implementation='flex'):
        """
        Builds the list of active mask functions and generates the final mask tensor.
        They can be appended to the final mask in "and" modality or "or" modality.
        """
        if kv is None:
            kv=query # If it's a self attention, ignore kv and use only query
        if implementation=='sdpa':
			if nested:
				print("WARNING: ATTENTION MASK NOT SUPPORTED IN SDPA + NESTED LAYOUT")
				return None
			else:
				causal_mask = torch.triu(torch.ones(L, S, device=device, dtype=torch.bool), diagonal=1)
				sliding_mask = torch.abs(torch.arange(L,device=device).unsqueeze(1) - torch.arange(S,device=device).unsqueeze(0)) <= window
				#Add the and/or + the document and cloze options.
				# Return the attention mask
		elif implementation=='flex':
			and_fns = []
			or_fns = []
			if 'causal' in attention_types:
				and_fns.append(self._mask_fn_causal)
			if is_sliding:
				bound_sliding_fn = partial(self._mask_fn_sliding, window_size=self.config.sliding_window_size)
				and_fns.append(bound_sliding_fn)

			if 'document' in attention_types:
				bound_doc_fn = partial(self._mask_fn_document, document_mask=document_mask)
				or_fns.append(bound_doc_fn)
			if 'cloze' in attention_types:
				bound_cloze_fn = partial(self._mask_fn_cloze, cloze_mask=cloze_mask)
				and_fns.append(bound_cloze_fn)
			combined_and_conditions = and_masks(*and_fns) if and_fns else None

			final_mask_logic = combined_and_conditions
			if or_fns:
				combined_or_conditions = or_masks(*or_fns)
				if final_mask_logic is not None and combined_or_conditions is not None:
					final_mask_logic = or_masks(final_mask_logic, combined_or_conditions)
				elif combined_or_conditions is not None :
					final_mask_logic = combined_or_conditions

			if not and_fns and not or_fns and final_mask_logic is None: 
				 pass 
			if nested:
				return create_nested_block_mask(
					mask_mod=final_mask_logic,
					B=batch_size, H=num_heads,
					q_nt=query, kv_nt=kv,
					BLOCK_SIZE=128,
				)
			else:
				return create_block_mask(
					mask_mod=final_mask_logic,
					B=batch_size, H=num_heads,
					Q_LEN=query.shape[-2], KV_LEN=kv.shape[-2],
					BLOCK_SIZE=128,
				) 

