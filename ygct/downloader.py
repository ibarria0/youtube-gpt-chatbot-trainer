import os
import concurrent.futures
import requests
import utils
from youtube_transcript_api import YouTubeTranscriptApi
from tqdm import tqdm

api_key = os.getenv('GOOGLE_API_KEY')

def get_channel_id(channel_name):
    # Use the YouTube Data API to search for the channel
    url = f"https://www.googleapis.com/youtube/v3/search?key={api_key}&q={channel_name}&part=snippet"
    response = requests.get(url)
    data = response.json()

    # Extract the channel ID from the API response
    channel_id = data["items"][0]["snippet"]["channelId"]
    return channel_id

def get_channel_videos(channel_id):
    # Use the YouTube Data API to get a list of the channel's videos
    url = f"https://www.googleapis.com/youtube/v3/search?key={api_key}&channelId={channel_id}&part=snippet,id&order=date&maxResults=2000"
    response = requests.get(url)
    data = response.json()

    # Extract the video URLs from the API response
    try:
        return [(item["id"]["videoId"], item["snippet"]["title"]) for item in data["items"]]
    except Exception as e:
        print(data)
        raise e

def get_transcript_for_video(video_id, title):
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    raw_transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'es'])
    sections = [(title, body, tokens) for body, tokens in utils.get_sections(raw_transcript)]
    return sections

def extract_content(channel):
    channel_id = get_channel_id(channel)
    print(f'channel id: {channel_id}')
    content = []
    videos = get_channel_videos(channel_id)
    print(f'got {len(videos)} videos')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_url = {executor.submit(get_transcript_for_video, video_id, title): video_id for video_id, title in videos[:1000]}
        for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(future_to_url)):
            video_id = future_to_url[future]
            try:
                sections = future.result()
                content.extend(sections)
            except Exception as exc:
                pass
    return content

