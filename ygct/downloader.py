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

def get_channel_videos(channel_id, page_token=None):
    # Use the YouTube Data API to get a list of the channel's videos
    url = f"https://www.googleapis.com/youtube/v3/search?key={api_key}&channelId={channel_id}&part=snippet,id&order=date&maxResults=20"
    if page_token is not None:
        url += f"&pageToken={page_token}"
    response = requests.get(url)
    data = response.json()


    # Extract the video URLs from the API response
    try:
        videos = [(item["id"]["videoId"], item["snippet"]["title"], item['snippet']['description']) for item in data["items"] if 'videoId' in item["id"]]
    except Exception as e:
        print(e)
        videos = []
        print(data)
        for item in data["items"]:
            print(item['id'])

    next_page_token = data.get("nextPageToken")
    return videos, next_page_token

def get_transcript_for_video(video_id):
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    raw_transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'es', 'en-US'])
    sections = [(body, tokens) for body, tokens in utils.get_sections(raw_transcript)]
    return sections

def get_all_videos_for_channel(channel_id):
    page_token = None
    videos = []
    while True:
        video_items, page_token = get_channel_videos(channel_id, page_token)
        videos.extend(video_items)
        if page_token is None:
            break
    print(f'got {len(videos)} videos')
    return videos

def extract_content(channel):
    """
    extract transcripts and descriptions from all videos intos sections and
    returns a list of (title, body, tokens)
    """

    content = []
    channel_id = get_channel_id(channel)
    print(f'channel id: {channel_id}')
    videos = get_all_videos_for_channel(channel_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(get_transcript_for_video, video_id): (video_id, title, description) for video_id, title, description in videos}
        for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(future_to_url)):
            video_id, title, description = future_to_url[future]
            try:
                sections = future.result()
                content.extend([(title, body, tokens) for body, tokens in sections])
                #inject description
                content.append((title, description, utils.count_tokens(description)))
            except Exception as exc:
                print(exc)
                pass
    return content
