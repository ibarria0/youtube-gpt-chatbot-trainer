import concurrent.futures
import utils
from youtube_transcript_api import YouTubeTranscriptApi
from tqdm import tqdm
import scrapetube

def get_transcript_for_video(video_id):
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    raw_transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'es', 'en-US'])
    sections = [(body, tokens) for body, tokens in utils.get_sections(raw_transcript)]
    return sections

def get_all_videos_for_channel(channel_url):
    videos = scrapetube.get_channel(channel_url=channel_url)

    # Extract the video URLs from the API response
    try:
        videos = [(item["videoId"], item["title"]["runs"][0]["text"]) for item in videos]
    except Exception as e:
        print(e)

    return videos

def extract_content(channel_url):
    """
    extract transcripts and descriptions from all videos intos sections and
    returns a list of (title, body, tokens)
    """

    content = []
    videos = get_all_videos_for_channel(channel_url)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(get_transcript_for_video, video_id): (video_id, title) for video_id, title in videos}
        for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(future_to_url)):
            video_id, title = future_to_url[future]
            try:
                sections = future.result()
                content.extend([(title, body, tokens) for body, tokens in sections])
            except Exception as exc:
                print(exc)
                pass
    return content
