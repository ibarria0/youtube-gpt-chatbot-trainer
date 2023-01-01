import os
import requests
from youtube_transcript_api import YouTubeTranscriptApi

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
    url = f"https://www.googleapis.com/youtube/v3/search?key={api_key}&channelId={channel_id}&part=snippet,id&order=date&maxResults=20"
    response = requests.get(url)
    data = response.json()

    # Extract the video URLs from the API response
    try:
        return [(item["id"]["videoId"], item["snippet"]["title"]) for item in data["items"]]
    except Exception as e:
        print(data)
        raise e

def get_transcript_for_video(video_id):
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    raw_transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'es'])
    return raw_transcript


