"""
Example client for TRELLIS API
"""

import requests
import time
from pathlib import Path


class TrellisClient:
    """Client for TRELLIS API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def convert_image(self, image_path: str, seed: int = 1, timeout: int = 300):
        """
        Convert image to 3D GLB file

        Args:
            image_path: Path to input image
            seed: Random seed for reproducibility
            timeout: Maximum wait time in seconds

        Returns:
            Path to downloaded GLB file
        """
        # Upload image
        print(f"Uploading: {image_path}")
        with open(image_path, 'rb') as f:
            files = {'file': f}
            params = {'seed': seed}
            response = requests.post(
                f"{self.base_url}/api/convert",
                files=files,
                params=params
            )
            response.raise_for_status()

        job_data = response.json()
        job_id = job_data['job_id']
        print(f"Job ID: {job_id}")
        print(f"Status: {job_data['status']}")

        # Poll for completion
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Job did not complete within {timeout} seconds")

            # Check status
            response = requests.get(f"{self.base_url}/api/status/{job_id}")
            response.raise_for_status()
            status_data = response.json()

            print(f"Status: {status_data['status']} - {status_data.get('message', '')}")

            if status_data['status'] == 'completed':
                # Download result
                print("Downloading result...")
                download_url = f"{self.base_url}{status_data['download_url']}"
                response = requests.get(download_url)
                response.raise_for_status()

                # Save file
                output_path = Path(f"output_{job_id}.glb")
                with open(output_path, 'wb') as f:
                    f.write(response.content)

                print(f"✓ Success! Saved to: {output_path}")
                return output_path

            elif status_data['status'] == 'failed':
                raise Exception(f"Job failed: {status_data.get('error', 'Unknown error')}")

            # Wait before next poll
            time.sleep(2)

    def get_status(self, job_id: str):
        """Get job status"""
        response = requests.get(f"{self.base_url}/api/status/{job_id}")
        response.raise_for_status()
        return response.json()

    def download(self, job_id: str, output_path: str = None):
        """Download result"""
        response = requests.get(f"{self.base_url}/api/download/{job_id}")
        response.raise_for_status()

        if output_path is None:
            output_path = f"output_{job_id}.glb"

        with open(output_path, 'wb') as f:
            f.write(response.content)

        return output_path

    def health(self):
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = TrellisClient("http://localhost:8000")

    # Check health
    print("Checking API health...")
    print(client.health())
    print()

    # Convert image
    image_path = "your-image.jpg"  # Change this
    if Path(image_path).exists():
        result = client.convert_image(image_path, seed=1)
        print(f"\nResult saved to: {result}")
    else:
        print(f"Image not found: {image_path}")
        print("\nExample API calls:")
        print("  client.convert_image('image.jpg', seed=1)")
        print("  client.get_status('job-id')")
        print("  client.download('job-id', 'output.glb')")
