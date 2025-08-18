import asyncio
import aiohttp
import time
import json
from datetime import datetime
import statistics

class LargeContextStressTest:
    def __init__(self, url, num_requests, concurrent_requests, input_tokens=10000, output_tokens=200):
        self.url = url
        self.num_requests = num_requests
        self.concurrent_requests = concurrent_requests
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.results = []
        self.errors = []
        
    def generate_large_prompt(self, token_count):
        """Generate a prompt with approximately the specified number of tokens"""
        # Rough estimate: 1 token ≈ 4 characters, so we need ~40,000 characters for 10k tokens
        # Average word is ~5 characters, so ~1.25 tokens per word
        
        base_text = """The following is a comprehensive analysis of various technological advances 
        and their implications for society, economics, and human development. 
        This document explores multiple dimensions of progress across different fields including 
        artificial intelligence, quantum computing, biotechnology, renewable energy, space exploration, 
        nanotechnology, robotics, telecommunications, and material science. 
        Each section provides detailed examination of current state, future projections, 
        challenges, opportunities, and interdisciplinary connections. """
        
        # Create a large context by repeating and varying content
        sections = [
            "In the field of artificial intelligence, we observe rapid advancement in neural network architectures, 
            transformer models, and multimodal learning systems. The implications extend beyond mere automation 
            to fundamental changes in how we process information, make decisions, and understand intelligence itself. ",
            
            "Quantum computing represents a paradigm shift in computational capabilities, offering exponential 
            speedup for certain problem classes. Current developments in quantum error correction, qubit coherence, 
            and quantum algorithms suggest we are approaching practical quantum advantage for real-world applications. ",
            
            "Biotechnology continues to revolutionize medicine through gene therapy, CRISPR technology, and 
            synthetic biology. The convergence of computational biology and laboratory techniques enables 
            unprecedented precision in understanding and manipulating biological systems. ",
            
            "The renewable energy sector demonstrates remarkable cost reductions and efficiency improvements 
            in solar photovoltaics, wind turbines, and energy storage systems. Grid integration challenges 
            and intermittency issues are being addressed through smart grid technologies and advanced forecasting. ",
            
            "Space exploration enters a new era with commercial spaceflight, asteroid mining prospects, and 
            Mars colonization plans. The reduction in launch costs and development of reusable rockets 
            fundamentally changes the economics of space access. ",
        ]
        
        # Build the large prompt
        large_prompt = base_text
        
        # Repeat sections to reach target token count
        # Roughly 750 tokens per full cycle through sections
        estimated_tokens = 0
        word_count = 0
        
        while estimated_tokens < token_count:
            for section in sections:
                large_prompt += section
                word_count += len(section.split())
                estimated_tokens = word_count * 1.25  # Rough token estimate
                
                if estimated_tokens >= token_count:
                    break
        
        # Add a specific question at the end
        large_prompt += f"\n\nGiven this extensive context of approximately {self.input_tokens} tokens, "
        large_prompt += "please provide a comprehensive summary that identifies the three most important "
        large_prompt += "technological trends and their potential convergence points. Focus on practical "
        large_prompt += "implications for the next decade."
        
        return large_prompt
    
    async def make_request(self, session, request_id):
        """Make a single request with large context"""
        
        # Generate large input prompt
        large_prompt = self.generate_large_prompt(self.input_tokens)
        
        payload = {
            "prompt": large_prompt,
            "max_tokens": self.output_tokens,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Request {request_id}: "
              f"Sending ~{self.input_tokens} input tokens...")
        
        start_time = time.time()
        try:
            async with session.post(self.url, json=payload, timeout=300) as response:
                result = await response.json()
                end_time = time.time()
                
                duration = end_time - start_time
                input_tokens = result.get('usage', {}).get('prompt_tokens', 0)
                output_tokens = result.get('usage', {}).get('completion_tokens', 0)
                total_tokens = input_tokens + output_tokens
                tokens_per_second = output_tokens / duration if duration > 0 else 0
                
                self.results.append({
                    'request_id': request_id,
                    'duration': duration,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': total_tokens,
                    'tokens_per_second': tokens_per_second,
                    'status': response.status
                })
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Request {request_id} completed: "
                      f"{duration:.2f}s, Input: {input_tokens}, Output: {output_tokens}, "
                      f"Speed: {tokens_per_second:.2f} tok/s")
                
                return True
        except asyncio.TimeoutError:
            self.errors.append({
                'request_id': request_id,
                'error': 'Timeout after 300 seconds',
                'duration': 300
            })
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Request {request_id} TIMEOUT")
            return False
        except Exception as e:
            self.errors.append({
                'request_id': request_id,
                'error': str(e),
                'duration': time.time() - start_time
            })
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Request {request_id} failed: {e}")
            return False
    
    async def run_test(self):
        """Run the stress test with large context"""
        print(f"\n{'='*70}")
        print(f"LARGE CONTEXT STRESS TEST")
        print(f"{'='*70}")
        print(f"Requests: {self.num_requests}")
        print(f"Concurrent: {self.concurrent_requests}")
        print(f"Input tokens per request: ~{self.input_tokens}")
        print(f"Output tokens per request: {self.output_tokens}")
        print(f"URL: {self.url}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        # Create connector with longer timeout
        timeout = aiohttp.ClientTimeout(total=300)
        connector = aiohttp.TCPConnector(limit=self.concurrent_requests)
        
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            # Process requests in batches
            for batch_start in range(0, self.num_requests, self.concurrent_requests):
                batch_size = min(self.concurrent_requests, self.num_requests - batch_start)
                tasks = []
                
                for i in range(batch_size):
                    request_id = batch_start + i
                    if request_id < self.num_requests:
                        tasks.append(self.make_request(session, request_id))
                
                await asyncio.gather(*tasks)
        
        total_duration = time.time() - start_time
        self.print_summary(total_duration)
    
    def print_summary(self, total_duration):
        """Print test summary"""
        print(f"\n{'='*70}")
        print(f"STRESS TEST SUMMARY")
        print(f"{'='*70}")
        
        if self.results:
            durations = [r['duration'] for r in self.results]
            input_tokens = [r['input_tokens'] for r in self.results]
            output_tokens = [r['output_tokens'] for r in self.results]
            tps = [r['tokens_per_second'] for r in self.results]
            
            print(f"Total test duration: {total_duration:.2f} seconds")
            print(f"Successful requests: {len(self.results)}/{self.num_requests}")
            print(f"Failed requests: {len(self.errors)}")
            
            print(f"\nResponse times (seconds):")
            print(f"  Min: {min(durations):.2f}")
            print(f"  Max: {max(durations):.2f}")
            print(f"  Mean: {statistics.mean(durations):.2f}")
            print(f"  Median: {statistics.median(durations):.2f}")
            
            print(f"\nTokens processed:")
            print(f"  Total input tokens: {sum(input_tokens):,}")
            print(f"  Total output tokens: {sum(output_tokens):,}")
            print(f"  Average input per request: {statistics.mean(input_tokens):.0f}")
            print(f"  Average output per request: {statistics.mean(output_tokens):.0f}")
            
            print(f"\nThroughput:")
            print(f"  Requests per second: {len(self.results)/total_duration:.3f}")
            print(f"  Output tokens/second (avg): {statistics.mean(tps):.2f}")
            print(f"  Total tokens processed: {sum(input_tokens) + sum(output_tokens):,}")
        
        if self.errors:
            print(f"\nErrors:")
            for error in self.errors[:5]:
                print(f"  Request {error['request_id']}: {error['error']}")
        
        # Save results
        with open('large_context_results.json', 'w') as f:
            json.dump({
                'summary': {
                    'total_duration': total_duration,
                    'num_requests': self.num_requests,
                    'concurrent_requests': self.concurrent_requests,
                    'target_input_tokens': self.input_tokens,
                    'successful': len(self.results),
                    'failed': len(self.errors)
                },
                'results': self.results,
                'errors': self.errors
            }, f, indent=2)
        
        print(f"\nDetailed results saved to large_context_results.json")

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Stress test with large input context')
    parser.add_argument('-n', '--num-requests', type=int, default=5,
                        help='Number of requests (default: 5)')
    parser.add_argument('-c', '--concurrent', type=int, default=2,
                        help='Concurrent requests (default: 2)')
    parser.add_argument('-i', '--input-tokens', type=int, default=10000,
                        help='Input tokens per request (default: 10000)')
    parser.add_argument('-o', '--output-tokens', type=int, default=200,
                        help='Max output tokens per request (default: 200)')
    
    args = parser.parse_args()
    
    tester = LargeContextStressTest(
        url='http://localhost:8000/v1/generate',
        num_requests=args.num_requests,
        concurrent_requests=args.concurrent,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens
    )
    
    await tester.run_test()

if __name__ == "__main__":
    asyncio.run(main())