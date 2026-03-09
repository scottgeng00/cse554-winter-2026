from chunked_engine import Engine, Request
import torch

class InputRequest:
    def __init__(self, input_str: str, output_len: int):
        self.input_str = input_str
        self.output_len = output_len
        
class Scheduler:
    def __init__(self, engine: Engine, token_batch_size: int):
        self.engine = engine
        self.token_batch_size = token_batch_size
        self.pending_input_req: list[InputRequest] = []
        self.decode_req: list[Request] = []
        self.scheduled_prefill_req: list[Request] = []
        self.completed: list[Request] = []
        self.unique_req_id: int = 0
        self.pending_prefill:list[Request] = []
    
    def add_req(self, input_req: InputRequest):
        self.pending_input_req.append(input_req)
        
    def finished(self) -> bool:
        return not self.pending_input_req and not self.decode_req and not self.pending_prefill

    def get_token_batch_size(self) -> int:
        sum = 0
        for req in self.scheduled_prefill_req:
            sum += req.scheduling_length
        for req in self.decode_req:
            sum += 1
        return sum

    def run(self):
        # Schedule new prefill requests until batch is full or no pending inputs
        
        while self.pending_input_req:
            pending_req = self.pending_input_req.pop(0)
            req_id = self.unique_req_id
            self.unique_req_id += 1
            prompt_ids = self.engine.tokenizer(
                pending_req.input_str, return_tensors="pt"
            ).input_ids[0]
            req = Request(
                req_id, prompt_ids, pending_req.output_len
            )
            self.pending_prefill.append(req)

        current_budget_used = self.get_token_batch_size()
        available_budget = self.token_batch_size - current_budget_used
        
        # Schedule prefill request and move it to the scheduled_prefill_req.
        # Limit the budget, and chunk the request as necessary.
        # If a request is chunked, keep it still in the pending prefill queue. Otherwise, remove from the queue.
        # Set tokens of the current chunk in the request scheduling_pf_tokens and set remaining_prefill_tokens properly
        while self.pending_prefill and available_budget > 0:
            pending_req = self.pending_prefill[0]
            start_idx = pending_req.prompt_length - pending_req.remaining_prefill_tokens
            # send everything
            if pending_req.remaining_prefill_tokens <= available_budget:
                pending_req.scheduling_pf_tokens = pending_req.prompt_token_ids[start_idx:]
                available_budget -= pending_req.remaining_prefill_tokens
                pending_req.remaining_prefill_tokens = 0
                self.pending_prefill.pop(0)
                pending_req.last_chunk = True
                self.scheduled_prefill_req.append(pending_req)
            else:  # then just throw as much in as possible.
                pending_req.scheduling_pf_tokens = pending_req.prompt_token_ids[start_idx:start_idx + available_budget]
                pending_req.remaining_prefill_tokens -= available_budget
                pending_req.last_chunk = False
                available_budget = 0
                self.scheduled_prefill_req.append(pending_req)
                break
            
        # Build the list of requests to send to the engine
        request_list_total = []
        decode_num = 0
        request_list_total = self.decode_req + self.scheduled_prefill_req
        decode_num = len(self.decode_req)
        # nothing left to do
        if not request_list_total:
            return
        new_tokens = self.engine.run(request_list_total, decode_num)

        # Append newly generated tokens to each request's output buffer
        # For prefill, only append if this cycle is the last chunk
        for idx, (req, new_token) in enumerate(zip(request_list_total, new_tokens)):
            # decode: always append
            # prefill: only append if this cycle is the last chunk
            if idx < decode_num:
                req.output_token_ids = torch.cat(
                    [req.output_token_ids, new_token.view(1)], dim=0
                )
            elif req.last_chunk:
                req.output_token_ids = torch.cat(
                    [req.output_token_ids, new_token.view(1)], dim=0
                )

        # Check which decode requests have finished
        ongoing_decode: list[Request] = []
        for req in self.decode_req:
            generated_tokens = req.current_length - req.prompt_length
            if generated_tokens >= req.output_length:
                self.completed.append(req)
                cache = self.engine.kv_cache_map.pop(req.request_id, None)
                if cache is not None:
                    cache.release()
            else:
                ongoing_decode.append(req)
        self.decode_req = ongoing_decode

        # Move scheduled prefill requests into decode queue
        for req in self.scheduled_prefill_req:
            if req.last_chunk:
                self.decode_req.append(req)
        self.scheduled_prefill_req = []
    
    def print_completed(self):
        for i, req in enumerate(self.completed):
            text = self.engine.tokenizer.decode(
                req.output_token_ids, skip_special_tokens=True
            )
            print(f"Id = {i}: {text}")
