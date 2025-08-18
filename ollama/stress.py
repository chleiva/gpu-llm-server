import asyncio
import aiohttp
import time
import json
import sys
import argparse
from datetime import datetime
import statistics

class StressTest:
    def __init__(self, url, num_requests, concurrent_requests, max_tokens=100):
        self.url = url
        self.num_requests = num_requests
        self.concurrent_requests = concurrent_requests
        self.max_tokens = max_tokens
        self.results = []
        self.errors = []
        
    async def make_request(self, session, request_id):
        """Make a single request to the API"""
        prompts = [
            "Explain quantum computing in simple terms",
            "Write a story about a robot learning to paint",
            "What are the benefits of renewable energy?",
            "Describe the process of machine learning",
            "How does blockchain technology work?",
            "Write a poem about artificial intelligence",
            "Explain the theory of relativity",
            "What is the future of space exploration?",
            "Describe the impact of social media on society",
            "How do neural networks learn?"
        ]
        
        prompt = prompts[request_id % len(prompts)]
        
        payload = {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": 0.7
        }
        
        start_time = time.time()
        try:
            async with session.post(self.url, json=payload) as response:
                result = await response.json()
                end_time = time.time()
                
                duration = end_time - start_time
                tokens = result.get('usage', {}).get('completion_tokens', 0)
                tokens_per_second = tokens / duration if duration > 0 else 0
                
                self.results.append({
                    'request_id': request_id,
                    'duration': duration,
                    'tokens': tokens,
                    'tokens_per_second': tokens_per_second,
                    'status': response.status
                })
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Request {request_id}: "
                      f"{duration:.2f}s, {tokens} tokens, {tokens_per_second:.2f} tok/s")
                
                return True
        except Exception as e:
            self.errors.append({
                'request_id': request_id,
                'error': str(e),
                'duration': time.time() - start_time
            })
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Request {request_id} failed: {e}")
            return False
    
    async def run_batch(self, batch_start, batch_size):
        """Run a batch of concurrent requests"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(batch_size):
                request_id = batch_start + i
                if request_id < self.num_requests:
                    tasks.append(self.make_request(session, request_id))
            
            return await asyncio.gather(*tasks)
    
    async def run_test(self):
        """Run the complete stress test"""
        print(f"\n{'='*60}")
        print(f"Starting stress test: {self.num_requests} requests, "
              f"{self.concurrent_requests} concurrent")
        print(f"URL: {self.url}")
        print(f"Max tokens per request: {self.max_tokens}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Process requests in batches
        for batch_start in range(0, self.num_requests, self.concurrent_requests):
            batch_size = min(self.concurrent_requests, self.num_requests - batch_start)
            await self.run_batch(batch_start, batch_size)
        
        total_duration = time.time() - start_time
        
        # Print summary
        self.print_summary(total_duration)
    
    def print_summary(self, total_duration):
        """Print test summary statistics"""
        print(f"\n{'='*60}")
        print("STRESS TEST SUMMARY")
        print(f"{'='*60}")
        
        if self.results:
            durations = [r['duration'] for r in self.results]
            tokens = [r['tokens'] for r in self.results]
            tps = [r['tokens_per_second'] for r in self.results]
            
            print(f"Total test duration: {total_duration:.2f} seconds")
            print(f"Successful requests: {len(self.results)}/{self.num_requests}")
            print(f"Failed requests: {len(self.errors)}")
            print(f"\nResponse times (seconds):")
            print(f"  Min: {min(durations):.2f}")
            print(f"  Max: {max(durations):.2f}")
            print(f"  Mean: {statistics.mean(durations):.2f}")
            print(f"  Median: {statistics.median(durations):.2f}")
            if len(durations) > 1:
                print(f"  Std Dev: {statistics.stdev(durations):.2f}")
            
            print(f"\nTokens generated:")
            print(f"  Total: {sum(tokens)}")
            print(f"  Average per request: {statistics.mean(tokens):.0f}")
            
            print(f"\nThroughput:")
            print(f"  Requests per second: {len(self.results)/total_duration:.2f}")
            print(f"  Average tokens/second: {statistics.mean(tps):.2f}")
            print(f"  Total tokens/second: {sum(tokens)/total_duration:.2f}")
        
        if self.errors:
            print(f"\nErrors encountered:")
            for error in self.errors[:5]:  # Show first 5 errors
                print(f"  Request {error['request_id']}: {error['error']}")
        
        # Save results to file
        with open('stress_test_results.json', 'w') as f:
            json.dump({
                'summary': {
                    'total_duration': total_duration,
                    'num_requests': self.num_requests,
                    'concurrent_requests': self.concurrent_requests,
                    'successful': len(self.results),
                    'failed': len(self.errors)
                },
                'results': self.results,
                'errors': self.errors
            }, f, indent=2)
        print(f"\nDetailed results saved to stress_test_results.json")

async def main():
    parser = argparse.ArgumentParser(description='Stress test the LLM API')
    parser.add_argument('-n', '--num-requests', type=int, default=20,
                        help='Total number of requests to make')
    parser.add_argument('-c', '--concurrent', type=int, default=5,
                        help='Number of concurrent requests')
    parser.add_argument('-t', '--tokens', type=int, default=100,
                        help='Max tokens per request')
    parser.add_argument('-u', '--url', type=str, 
                        default='http://localhost:8000/v1/generate',
                        help='API endpoint URL')
    
    args = parser.parse_args()
    
    tester = StressTest(
        url=args.url,
        num_requests=args.num_requests,
        concurrent_requests=args.concurrent,
        max_tokens=args.tokens
    )
    
    await tester.run_test()

if __name__ == "__main__":
    asyncio.run(main())