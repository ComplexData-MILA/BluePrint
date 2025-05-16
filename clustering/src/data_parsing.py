import os
import json
import glob
from tqdm import tqdm
from collections import defaultdict
from utils import logger, DATA_ROOT, default_user_data
import numpy as np

class DataParser:
    def __init__(self, embedding_manager):
        self.embedding_manager = embedding_manager
        self.user_data = defaultdict(default_user_data)
    
    def parse_bluesky_files(self, start_date=2, end_date=28, n_workers=5):
        """Parse all Bluesky files and organize by user using multiprocessing"""
        from multiprocessing import Pool, cpu_count
        
        # Create a pool of workers
        num_cpus = min(cpu_count(), end_date - start_date + 1, n_workers)
        pool = Pool(processes=num_cpus)
        
        logger.info(f"Starting multiprocessing pool with {num_cpus} workers (CPUs: {cpu_count()})")
        
        # Process each day in parallel
        results = pool.map(self.process_day, range(start_date, end_date + 1))
        
        # Close the pool
        pool.close()
        pool.join()
        
        # Merge results
        for result in results:
            if result is not None:
                user_data, embedding_cache = result
                
                # Merge user data
                for user_id, data in user_data.items():
                    self.user_data[user_id]['posts'].extend(data['posts'])
                    self.user_data[user_id]['replies'].extend(data['replies'])
                    self.user_data[user_id]['likes'].extend(data['likes'])
                    self.user_data[user_id]['unlikes'].extend(data['unlikes'])
                    self.user_data[user_id]['reposts'].extend(data['reposts'])
                    self.user_data[user_id]['unreposts'].extend(data['unreposts'])
                    self.user_data[user_id]['follows'].extend(data['follows'])
                    self.user_data[user_id]['unfollows'].extend(data['unfollows'])
                    self.user_data[user_id]['blocks'].extend(data['blocks'])
                    self.user_data[user_id]['unblocks'].extend(data['unblocks'])
                    self.user_data[user_id]['post_updates'].extend(data['post_updates'])
                    self.user_data[user_id]['post_deletes'].extend(data['post_deletes'])
                    self.user_data[user_id]['quotes'].extend(data['quotes'])
                    self.user_data[user_id]['post_embeddings'].extend(data['post_embeddings'])
                    # Update profile if needed
                    if data['profile'] is not None:
                        self.user_data[user_id]['profile'] = data['profile']
                
                # Merge embedding cache
                self.embedding_manager.update_cache(embedding_cache)
        
        # Filter users with insufficient English content
        self.filter_users()
        
        # Save cache after processing all files
        logger.info("Saving merged data from all processes")
        self.embedding_manager.save_cache()
    
    def process_day(self, day):
        """Process all files for a single day"""
        import os
        from embedding import EmbeddingManager
        from collections import defaultdict
        
        day_str = f"{day:02d}"
        day_path = os.path.join(os.path.expanduser(DATA_ROOT), day_str)
        
        # Create a new EmbeddingManager for this process
        embedding_manager = EmbeddingManager()
        
        # Create new data structures for this process
        process_user_data = defaultdict(default_user_data)
        
        if not os.path.exists(day_path):
            logger.warning(f"Path does not exist: {day_path}")
            return None
        
        for hour in range(24):
            hour_str = f"{hour:02d}"
            hour_path = os.path.join(day_path, hour_str)
            
            if not os.path.exists(hour_path):
                continue
            
            logger.info(f"Process {os.getpid()} processing data for {day_str}/{hour_str}")
            files = glob.glob(os.path.join(hour_path, "*"))
            
            for file_path in tqdm(files, desc=f"Day {day_str} Hour {hour_str}"):
                self.process_file(file_path, process_user_data, embedding_manager)
        
        return process_user_data, embedding_manager.get_cache()
    
    def process_file(self, file_path, user_data=None, embedding_manager=None):
        """Process a single Bluesky data file"""
        if user_data is None:
            user_data = self.user_data
        
        if embedding_manager is None:
            embedding_manager = self.embedding_manager
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        self.process_entry(data, user_data, embedding_manager)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
    
    def process_entry(self, data, user_data=None, embedding_manager=None):
        """Process a single data entry and categorize by type"""
        if user_data is None:
            user_data = self.user_data
            
        if embedding_manager is None:
            embedding_manager = self.embedding_manager
            
        try:
            if 'kind' not in data or data['kind'] != 'commit' or 'commit' not in data:
                return
            
            commit = data['commit']
            user_id = data['did']
            unix_epoch = data['time_us']
            collection = commit.get('collection', '')
            operation = commit.get('operation', '')
            record = commit.get('record', {})
            
            # Process based on collection type and operation
            if collection == 'app.bsky.feed.post':
                if operation == 'create':
                    self.process_post(user_id, record, commit, user_data, embedding_manager, unix_epoch)
                elif operation == 'update':
                    self.process_post_update(user_id, record, commit, user_data, embedding_manager, unix_epoch)
                elif operation == 'delete':
                    self.process_post_delete(user_id, commit, user_data, unix_epoch)
            elif collection == 'app.bsky.feed.like':
                if operation == 'create':
                    self.process_like(user_id, record, commit, user_data, unix_epoch)
                elif operation == 'delete':
                    self.process_unlike(user_id, commit, user_data, unix_epoch)
                else:
                    logger.warning(f"Unknown operation for like: {operation}")
            elif collection == 'app.bsky.feed.repost':
                if operation == 'create':
                    self.process_repost(user_id, record, commit, user_data, unix_epoch)
                elif operation == 'delete':
                    self.process_unrepost(user_id, commit, user_data, unix_epoch)
            elif collection == 'app.bsky.graph.follow':
                if operation == 'create':
                    self.process_follow(user_id, record, commit, user_data, unix_epoch)
                elif operation == 'delete':
                    self.process_unfollow(user_id, commit, user_data, unix_epoch)
                else:
                    logger.warning(f"Unknown operation for follow: {operation}")
            elif collection == 'app.bsky.graph.block':
                if operation == 'create':
                    self.process_block(user_id, record, commit, user_data, unix_epoch)
                elif operation == 'delete':
                    self.process_unblock(user_id, commit, user_data, unix_epoch)
            elif collection == 'app.bsky.actor.profile' and (operation == 'create' or operation == 'update'):
                self.process_profile(user_id, record, commit, user_data, unix_epoch)
        except Exception as e:
            logger.warning(f"Error processing entry: {str(e)}")
    
    def process_post(self, user_id, record, commit, user_data=None, embedding_manager=None, unix_epoch=0):
        """Process a post entry"""
        if user_data is None:
            user_data = self.user_data
            
        if embedding_manager is None:
            embedding_manager = self.embedding_manager
            
        if '$type' not in record or record['$type'] != 'app.bsky.feed.post':
            return
        
        text = record.get('text', '')
        created_at = record.get('createdAt', '')
        cid = commit.get('cid', '')
        rkey = commit.get('rkey', '')
        
        # Check language
        langs = record.get('langs', [])
        
        is_english = langs == ['en']
        
        # Check if it's a reply
        is_reply = 'reply' in record
        
        # Check if it's a quote
        is_quote = record.get('embed', {}).get('record', False)
        
        post_data = {
            'text': text,
            'created_at': created_at,
            'unix_epoch': unix_epoch,
            'cid': cid,
            'rkey': rkey,
            'uri': f"at://{user_id}/app.bsky.feed.post/{rkey}",
            'langs': langs,
            'is_english': is_english
        }
        
        # Add embedding if it's English
        if is_english:
            embedding = embedding_manager.get_embedding(text)
            post_data['embedding'] = embedding
            user_data[user_id]['post_embeddings'].append(embedding)
        
        if is_quote:
            post_data['quoted'] = record['embed']['record']
            user_data[user_id]['quotes'].append(post_data)
        elif is_reply:
            reply_data = record['reply']
            post_data['parent'] = reply_data.get('parent', {})
            post_data['root'] = reply_data.get('root', {})
            user_data[user_id]['replies'].append(post_data)
        else:
            user_data[user_id]['posts'].append(post_data)
    
    def process_post_update(self, user_id, record, commit, user_data=None, embedding_manager=None, unix_epoch=0):
        """Process a post update entry"""
        if user_data is None:
            user_data = self.user_data
        
        if embedding_manager is None:
            embedding_manager = self.embedding_manager
            
        if '$type' not in record or record['$type'] != 'app.bsky.feed.post':
            return
        
        text = record.get('text', '')
        created_at = record.get('createdAt', '')
        cid = commit.get('cid', '')
        rkey = commit.get('rkey', '')
        
        update_data = {
            'text': text,
            'created_at': created_at,
            'unix_epoch': unix_epoch,
            'cid': cid,
            'rkey': rkey,
            'uri': f"at://{user_id}/app.bsky.feed.post/{rkey}"
        }
        
        user_data[user_id]['post_updates'].append(update_data)
    
    def process_post_delete(self, user_id, commit, user_data=None, unix_epoch=0):
        """Process a post delete entry"""
        if user_data is None:
            user_data = self.user_data
        
        rkey = commit.get('rkey', '')
        
        delete_data = {
            'unix_epoch': unix_epoch,
            'rkey': rkey,
            'uri': f"at://{user_id}/app.bsky.feed.post/{rkey}"
        }
        
        user_data[user_id]['post_deletes'].append(delete_data)
    
    def process_like(self, user_id, record, commit, user_data=None, unix_epoch=0):
        """Process a like entry"""
        if user_data is None:
            user_data = self.user_data
            
        if '$type' not in record or record['$type'] != 'app.bsky.feed.like':
            return
        
        subject = record.get('subject', {})
        created_at = record.get('createdAt', '')
        
        like_data = {
            'subject': subject,
            'created_at': created_at,
            'unix_epoch': unix_epoch,
            'uri': subject.get('uri', ''),
            'cid': subject.get('cid', ''),
            'rkey': commit.get('rkey', '')
        }
        
        user_data[user_id]['likes'].append(like_data)
    
    def process_unlike(self, user_id, commit, user_data=None, unix_epoch=0):
        """Process an unlike entry"""
        if user_data is None:
            user_data = self.user_data
        
        rkey = commit.get('rkey', '')
        
        unlike_data = {
            'unix_epoch': unix_epoch,
            'rkey': rkey,
            'uri': f"at://{user_id}/app.bsky.feed.like/{rkey}"
        }
        
        user_data[user_id]['unlikes'].append(unlike_data)
    
    def process_repost(self, user_id, record, commit, user_data=None, unix_epoch=0):
        """Process a repost entry"""
        if user_data is None:
            user_data = self.user_data
            
        if '$type' not in record or record['$type'] != 'app.bsky.feed.repost':
            return
        
        subject = record.get('subject', {})
        created_at = record.get('createdAt', '')
        
        repost_data = {
            'subject': subject,
            'created_at': created_at,
            'unix_epoch': unix_epoch,
            'uri': subject.get('uri', ''),
            'cid': subject.get('cid', ''),
            'rkey': commit.get('rkey', '')
        }
        
        user_data[user_id]['reposts'].append(repost_data)
    
    def process_unrepost(self, user_id, commit, user_data=None, unix_epoch=0):
        """Process an unrepost entry"""
        if user_data is None:
            user_data = self.user_data
        
        rkey = commit.get('rkey', '')
        
        unrepost_data = {
            'unix_epoch': unix_epoch,
            'rkey': rkey,
            'uri': f"at://{user_id}/app.bsky.feed.repost/{rkey}"
        }
        
        user_data[user_id]['unreposts'].append(unrepost_data)
    
    def process_follow(self, user_id, record, commit, user_data=None, unix_epoch=0):
        """Process a follow entry"""
        if user_data is None:
            user_data = self.user_data
            
        if '$type' not in record or record['$type'] != 'app.bsky.graph.follow':
            return
        
        subject = record.get('subject', '')
        created_at = record.get('createdAt', '')
        
        follow_data = {
            'unix_epoch': unix_epoch,
            'subject': subject,  # The user being followed
            'created_at': created_at,
            'rkey': commit.get('rkey', ''),
        }
        
        user_data[user_id]['follows'].append(follow_data)
    
    def process_unfollow(self, user_id, commit, user_data=None, unix_epoch=0):
        """Process an unfollow entry"""
        if user_data is None:
            user_data = self.user_data
        
        rkey = commit.get('rkey', '')
        
        unfollow_data = {
            'unix_epoch': unix_epoch,
            'rkey': rkey,
            'uri': f"at://{user_id}/app.bsky.graph.follow/{rkey}"
        }
        
        user_data[user_id]['unfollows'].append(unfollow_data)
    
    def process_block(self, user_id, record, commit, user_data=None, unix_epoch=0):
        """Process a block entry"""
        if user_data is None:
            user_data = self.user_data
            
        if '$type' not in record or record['$type'] != 'app.bsky.graph.block':
            return
        
        subject = record.get('subject', '')
        created_at = record.get('createdAt', '')
        
        block_data = {
            'unix_epoch': unix_epoch,
            'subject': subject,  # The user being blocked
            'created_at': created_at,
            'rkey': commit.get('rkey', ''),
        }
        
        user_data[user_id]['blocks'].append(block_data)
    
    def process_unblock(self, user_id, commit, user_data=None, unix_epoch=0):
        """Process an unblock entry"""
        if user_data is None:
            user_data = self.user_data
        
        rkey = commit.get('rkey', '')
        
        unblock_data = {
            'unix_epoch': unix_epoch,
            'rkey': rkey,
            'uri': f"at://{user_id}/app.bsky.graph.block/{rkey}"
        }
        
        user_data[user_id]['unblocks'].append(unblock_data)
    
    def process_profile(self, user_id, record, commit, user_data=None, unix_epoch=0):
        """Process a profile entry"""
        if user_data is None:
            user_data = self.user_data
            
        if '$type' not in record or record['$type'] != 'app.bsky.actor.profile':
            return

        profile_data = {
            'display_name': record.get('displayName', ''),
            'description': record.get('description', ''),
            'created_at': commit.get('createdAt', ''),
            'unix_epoch': unix_epoch,
        }
        
        user_data[user_id]['profile'] = profile_data
    
    def filter_users(self):
        """Filter out non-English users and users with only a single post/reply"""
        filtered_users = {}
        
        for user_id, data in self.user_data.items():
            english_posts = [p for p in data['posts'] if p.get('is_english', False)]
            english_replies = [r for r in data['replies'] if r.get('is_english', False)]
            
            # Keep user if they have at least 2 English posts/replies
            if len(english_posts) + len(english_replies) >= 2:
                filtered_data = data.copy()
                filtered_data['posts'] = english_posts
                filtered_data['replies'] = english_replies
                filtered_users[user_id] = filtered_data
        
        logger.info(f"Filtered from {len(self.user_data)} to {len(filtered_users)} users with sufficient English content")
        self.user_data = defaultdict(default_user_data)
        
        # Update our user_data with filtered data
        for user_id, data in filtered_users.items():
            self.user_data[user_id] = data
    
    def compute_user_embeddings(self):
        """Compute average embeddings for each user"""
        user_embeddings = {}
        
        for user_id, data in self.user_data.items():
            # Combine posts and replies embeddings
            all_embeddings = []
            
            # Add post embeddings
            for post in data['posts']:
                if 'embedding' in post:
                    all_embeddings.append(post['embedding'])
            
            # Add reply embeddings
            for reply in data['replies']:
                if 'embedding' in reply:
                    all_embeddings.append(reply['embedding'])
            
            if all_embeddings:
                # Compute average embedding
                avg_embedding = np.mean(all_embeddings, axis=0)
                user_embeddings[user_id] = avg_embedding
        
        return user_embeddings