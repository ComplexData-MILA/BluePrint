import os
import json
import numpy as np
from tqdm import tqdm
from utils import logger, default_actions

class OutputFormatter:
    def __init__(self, cap_ignored_messages=1000):
        self.cap_ignored_messages = cap_ignored_messages
        self.output_dir = None
    
    def set_output_dir(self, output_dir):
        """Set the output directory for saving files."""
        self.output_dir = output_dir
        logger.info(f"OutputFormatter output directory set to: {self.output_dir}")

    def compile_cluster_data(self, cluster_users, ignored_content_by_cluster, user_data):
        """Compile data for each cluster into required format"""
        if not self.output_dir:
            logger.error("Output directory not set in OutputFormatter. Cannot compile cluster data.")
            return
            
        for cluster_id, users in cluster_users.items():
            logger.info(f"Compiling data for cluster {cluster_id} with {len(users)} users")
            
            # Create output file path using self.output_dir
            output_file = os.path.join(self.output_dir, f"cluster_{cluster_id}.jsonl")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                self.write_conversation_chains(f, users, user_data)
                self.write_like_actions(f, users, user_data)
                self.write_unlike_actions(f, users, user_data)
                self.write_repost_actions(f, users, user_data)
                self.write_unrepost_actions(f, users, user_data)
                self.write_follow_actions(f, users, user_data)
                self.write_unfollow_actions(f, users, user_data)
                self.write_block_actions(f, users, user_data)
                self.write_unblock_actions(f, users, user_data)
                self.write_post_update_actions(f, users, user_data)
                self.write_post_delete_actions(f, users, user_data)
                self.write_post_quote_actions(f, users, user_data)

                if cluster_id in ignored_content_by_cluster:
                    self.write_ignored_content(f, users, ignored_content_by_cluster[cluster_id])
    
    def write_conversation_chains(self, file, users, user_data):
        """Write conversation chains to output file"""
        # Collect all posts and replies from users in the cluster
        all_posts = {}
        reply_chains = {}
        
        # First, index all posts by URI for quick lookup
        for user_id in users:
            user_data_item = user_data[user_id]
            
            # Add posts
            for post in user_data_item['posts']:
                uri = post.get('uri', '')
                if uri:
                    post_data = {
                        'uri': uri,
                        'text': post.get('text', ''),
                        'user_id': user_id,
                        'created_at': post.get('created_at', '')
                    }
                    all_posts[uri] = post_data
            
            # Add replies
            for reply in user_data_item['replies']:
                uri = reply.get('uri', '')
                if uri:
                    reply_data = {
                        'uri': uri,
                        'text': reply.get('text', ''),
                        'user_id': user_id,
                        'created_at': reply.get('created_at', ''),
                        'parent_uri': reply.get('parent', {}).get('uri', ''),
                        'root_uri': reply.get('root', {}).get('uri', '')
                    }
                    all_posts[uri] = reply_data
                    
                    # Add to reply chains
                    parent_uri = reply_data['parent_uri']
                    if parent_uri:
                        if parent_uri not in reply_chains:
                            reply_chains[parent_uri] = []
                        reply_chains[parent_uri].append(uri)
        
        # Build conversation chains
        conversation_chains = []
        processed_uris = set()
        
        # Process each post from users in the cluster
        for user_id in users:
            user_data_item = user_data[user_id]
            
            # Process posts
            for post in user_data_item['posts']:
                uri = post.get('uri', '')
                if not uri or uri in processed_uris:
                    continue
                
                # Start a new chain
                chain = [{'user_id': user_id}]
                if post.get('text', ''):
                    chain[0]['text'] = post.get('text', '')
                chain[0]['unix_epoch'] = post.get('unix_epoch', 0)
                processed_uris.add(uri)
                
                # Add replies to this post
                self.add_replies_to_chain(chain, uri, reply_chains, all_posts, processed_uris, 0)
                
                if len(chain) > 0:
                    conversation_chains.append(chain)
            
            # Process replies (that might start a chain we care about)
            for reply in user_data_item['replies']:
                uri = reply.get('uri', '')
                if not uri or uri in processed_uris:
                    continue
                
                parent_uri = reply.get('parent', {}).get('uri', '')
                root_uri = reply.get('root', {}).get('uri', '')
                
                # If parent already processed, skip
                if parent_uri in processed_uris:
                    continue
                
                # If this is a reply to something not in our dataset
                # Start chain with this reply
                if not parent_uri or parent_uri not in all_posts:
                    chain = [{'user_id': user_id}]
                    if reply.get('text', ''):
                        chain[0]['text'] = reply.get('text', '')
                    chain[0]['unix_epoch'] = reply.get('unix_epoch', 0)
                    processed_uris.add(uri)
                    
                    # Add replies to this reply
                    self.add_replies_to_chain(chain, uri, reply_chains, all_posts, processed_uris, 0)
                    
                    if len(chain) > 0:
                        conversation_chains.append(chain)
                    continue
                
                # Try to reconstruct the conversation chain from root
                if root_uri and root_uri in all_posts and root_uri != parent_uri:
                    # This is a reply in a longer conversation
                    # Try to start from root if we haven't processed it yet
                    if root_uri not in processed_uris:
                        root_post = all_posts[root_uri]
                        chain = [{'user_id': root_post.get('user_id')}]
                        if root_post.get('text', ''):
                            chain[0]['text'] = root_post.get('text', '')
                        chain[0]['unix_epoch'] = root_post.get('unix_epoch', 0)
                        processed_uris.add(root_uri)
                        
                        # Reconstruct the conversation chain
                        self.add_replies_to_chain(chain, root_uri, reply_chains, all_posts, processed_uris, 0)
                        
                        if len(chain) > 0:
                            conversation_chains.append(chain)
                else:
                    # Start from parent post
                    parent_post = all_posts[parent_uri]
                    chain = [{'user_id': parent_post.get('user_id')}]
                    if parent_post.get('text', ''):
                        chain[0]['text'] = parent_post.get('text', '')
                    chain[0]['unix_epoch'] = parent_post.get('unix_epoch', 0)
                    processed_uris.add(parent_uri)
                    
                    # Add this reply
                    reply_msg = {'user_id': user_id}
                    if reply.get('text', ''):
                        reply_msg['text'] = reply.get('text', '')
                    reply_msg['unix_epoch'] = reply.get('unix_epoch', 0)
                    chain.append(reply_msg)
                    processed_uris.add(uri)
                    
                    # Add further replies
                    self.add_replies_to_chain(chain, uri, reply_chains, all_posts, processed_uris, 0)
                    
                    if len(chain) > 0:
                        conversation_chains.append(chain)
        
        # Write conversation chains to file
        for chain in conversation_chains:
            if len(chain) > 0:
                file.write(json.dumps(chain) + '\n')
    
    def add_replies_to_chain(self, chain, parent_uri, reply_chains, all_posts, processed_uris, depth):
        """Recursively add replies to a conversation chain"""
        # Limit reply depth to prevent huge chains
        if depth > 5:
            return
        
        # Get replies to this post/reply
        reply_uris = reply_chains.get(parent_uri, [])
        
        for reply_uri in reply_uris:
            # Skip if already processed
            if reply_uri in processed_uris:
                continue
                
            # Add this reply to the chain
            reply_data = all_posts.get(reply_uri, {})
            if not reply_data:
                continue
                
            reply_msg = {'user_id': reply_data.get('user_id')}
            if reply_data.get('text', ''):
                reply_msg['text'] = reply_data.get('text', '')
            reply_msg['unix_epoch'] = reply_data.get('unix_epoch', 0)
            chain.append(reply_msg)
            processed_uris.add(reply_uri)
            
            # Recursively add replies to this reply
            self.add_replies_to_chain(chain, reply_uri, reply_chains, all_posts, processed_uris, depth + 1)
    
    def write_like_actions(self, file, users, user_data):
        """Write like actions to output file"""
        for user_id in users:
            user_data_item = user_data[user_id]
            
            for like in user_data_item['likes']:
                uri = like.get('uri', '')
                
                # Find the original post that was liked
                post_content = self.find_content_by_uri(uri, user_data)
                
                if post_content and 'user_id' in post_content:
                    # Create the like action
                    content_msg = {'user_id': post_content.get('user_id')}
                    if post_content.get('text', ''):
                        content_msg['text'] = post_content.get('text', '')
                    content_msg['unix_epoch'] = post_content.get('unix_epoch', 0)
                    
                    action = default_actions()
                    action['like'] = True
                    action_msg = {
                        'user_id': user_id,
                        'actions': action,
                        'unix_epoch': like.get('unix_epoch', 0)
                    }
                    
                    action = [content_msg, action_msg]
                    file.write(json.dumps(action) + '\n')
    
    def write_unlike_actions(self, file, users, user_data):
        """Write unlike actions to output file"""
        for user_id in users:
            user_data_item = user_data[user_id]
            
            for unlike in user_data_item['unlikes']:
                rkey = unlike.get('rkey', '')
                
                # Try to find the original like to get the post URI
                original_like = None
                for like in user_data_item['likes']:
                    if like.get('rkey', '') == rkey:
                        original_like = like
                        break
                
                if not original_like:
                    continue
                
                uri = original_like.get('uri', '')
                
                # Find the original post that was unliked
                post_content = self.find_content_by_uri(uri, user_data)
                
                if post_content and 'user_id' in post_content:
                    # Create the unlike action
                    content_msg = {'user_id': post_content.get('user_id')}
                    if post_content.get('text', ''):
                        content_msg['text'] = post_content.get('text', '')
                    content_msg['unix_epoch'] = post_content.get('unix_epoch', 0)
                    
                    action = default_actions()
                    action['unlike'] = True
                    action_msg = {
                        'user_id': user_id,
                        'actions': action,
                        'unix_epoch': unlike.get('unix_epoch', 0)
                    }
                    
                    action = [content_msg, action_msg]
                    file.write(json.dumps(action) + '\n')
    
    def write_repost_actions(self, file, users, user_data):
        """Write repost actions to output file"""
        for user_id in users:
            user_data_item = user_data[user_id]
            
            for repost in user_data_item['reposts']:
                uri = repost.get('uri', '')
                
                # Find the original post that was reposted
                post_content = self.find_content_by_uri(uri, user_data)
                
                if post_content and 'user_id' in post_content:
                    # Create the repost action
                    content_msg = {'user_id': post_content.get('user_id')}
                    if post_content.get('text', ''):
                        content_msg['text'] = post_content.get('text', '')
                    content_msg['unix_epoch'] = post_content.get('unix_epoch', 0)
                    
                    action = default_actions()
                    action['repost'] = True
                    action_msg = {
                        'user_id': user_id,
                        'actions': action,
                        'unix_epoch': repost.get('unix_epoch', 0)
                    }
                    
                    action = [content_msg, action_msg]
                    file.write(json.dumps(action) + '\n')
    
    def write_unrepost_actions(self, file, users, user_data):
        """Write unrepost actions to output file"""
        for user_id in users:
            user_data_item = user_data[user_id]
            
            for unrepost in user_data_item['unreposts']:
                rkey = unrepost.get('rkey', '')
                
                # Try to find the original repost to get the post URI
                original_repost = None
                for repost in user_data_item['reposts']:
                    if repost.get('rkey', '') == rkey:
                        original_repost = repost
                        break
                
                if not original_repost:
                    continue
                
                uri = original_repost.get('uri', '')
                
                # Find the original post that was unreposted
                post_content = self.find_content_by_uri(uri, user_data)
                
                if post_content and 'user_id' in post_content:
                    # Create the unrepost action
                    content_msg = {'user_id': post_content.get('user_id')}
                    if post_content.get('text', ''):
                        content_msg['text'] = post_content.get('text', '')
                    content_msg['unix_epoch'] = post_content.get('unix_epoch', 0)
                    
                    action = default_actions()
                    action['unrepost'] = True
                    action_msg = {
                        'user_id': user_id,
                        'actions': action,
                        'unix_epoch': unrepost.get('unix_epoch', 0)
                    }
                    
                    action = [content_msg, action_msg]
                    file.write(json.dumps(action) + '\n')
    
    def write_follow_actions(self, file, users, user_data):
        """Write follow actions to output file"""
        for user_id in users:
            user_data_item = user_data[user_id]
            
            for follow in user_data_item['follows']:
                followed_user = follow.get('subject', '')
                
                if not followed_user or followed_user not in user_data:
                    continue
                
                # Find the most recent post from the followed user at time of follow
                followed_data = user_data[followed_user]
                
                # Combine posts and replies for the followed user
                all_content = followed_data['posts'] + followed_data['replies']
                
                # Sort by created_at (most recent first)
                all_content.sort(key=lambda x: x.get('unix_epoch', 0), reverse=True)
                
                # Get the follow timestamp
                follow_time = follow.get('unix_epoch', 0)
                
                # Find most recent post before the follow
                most_recent_post = None
                for content in all_content:
                    if content.get('unix_epoch', 0) <= follow_time:
                        most_recent_post = content
                        break
                
                # If found, write the follow action
                if most_recent_post:
                    content_msg = {'user_id': followed_user}
                    if most_recent_post.get('text', ''):
                        content_msg['text'] = most_recent_post.get('text', '')
                    content_msg['unix_epoch'] = most_recent_post.get('unix_epoch', 0)
                    
                    action = default_actions()
                    action['follow'] = True
                    action_msg = {
                        'user_id': user_id,
                        'actions': action,
                        'unix_epoch': follow.get('unix_epoch', 0)
                    }
                    
                    action = [content_msg, action_msg]
                    file.write(json.dumps(action) + '\n')
    
    def write_unfollow_actions(self, file, users, user_data):
        """Write unfollow actions to output file"""
        for user_id in users:
            user_data_item = user_data[user_id]
            
            for unfollow in user_data_item['unfollows']:
                rkey = unfollow.get('rkey', '')
                
                # Try to find the original follow to get the followed user
                original_follow = None
                for follow in user_data_item['follows']:
                    if follow.get('rkey', '') == rkey:
                        original_follow = follow
                        break
                
                if not original_follow:
                    continue
                
                followed_user = original_follow.get('subject', '')
                
                if not followed_user or followed_user not in user_data:
                    continue
                
                # Find the most recent post from the unfollowed user
                followed_data = user_data[followed_user]
                
                # Combine posts and replies for the followed user
                all_content = followed_data['posts'] + followed_data['replies']
                
                # Sort by created_at (most recent first)
                all_content.sort(key=lambda x: x.get('unix_epoch', 0), reverse=True)
                
                # Get the unfollow timestamp
                unfollow_time = unfollow.get('unix_epoch', 0)
                
                # Find most recent post before the unfollow
                most_recent_post = None
                for content in all_content:
                    if content.get('unix_epoch', 0) <= unfollow_time:
                        most_recent_post = content
                        break
                
                # If found, write the unfollow action
                if most_recent_post:
                    content_msg = {'user_id': followed_user}
                    if most_recent_post.get('text', ''):
                        content_msg['text'] = most_recent_post.get('text', '')
                    content_msg['unix_epoch'] = most_recent_post.get('unix_epoch', 0)
                    
                    action = default_actions()
                    action['unfollow'] = True
                    action_msg = {
                        'user_id': user_id,
                        'actions': action,
                        'unix_epoch': unfollow.get('unix_epoch', 0)
                    }
                    
                    action = [content_msg, action_msg]
                    file.write(json.dumps(action) + '\n')
    
    def write_block_actions(self, file, users, user_data):
        """Write block actions to output file"""
        for user_id in users:
            user_data_item = user_data[user_id]
            
            for block in user_data_item['blocks']:
                blocked_user = block.get('subject', '')
                
                if not blocked_user or blocked_user not in user_data:
                    continue
                
                # Find the most recent post from the blocked user
                blocked_data = user_data[blocked_user]
                
                # Combine posts and replies for the blocked user
                all_content = blocked_data['posts'] + blocked_data['replies']
                
                # Sort by created_at (most recent first)
                all_content.sort(key=lambda x: x.get('unix_epoch', 0), reverse=True)
                
                # Get the block timestamp
                block_time = block.get('unix_epoch', 0)
                
                # Find most recent post before the block
                most_recent_post = None
                for content in all_content:
                    if content.get('unix_epoch', 0) <= block_time:
                        most_recent_post = content
                        break
                
                # If found, write the block action
                if most_recent_post:
                    content_msg = {'user_id': blocked_user}
                    if most_recent_post.get('text', ''):
                        content_msg['text'] = most_recent_post.get('text', '')
                    content_msg['unix_epoch'] = most_recent_post.get('unix_epoch', 0)
                    
                    action = default_actions()
                    action['block'] = True
                    action_msg = {
                        'user_id': user_id,
                        'actions': action,
                        'unix_epoch': block.get('unix_epoch', 0)
                    }
                    
                    action = [content_msg, action_msg]
                    file.write(json.dumps(action) + '\n')
    
    def write_unblock_actions(self, file, users, user_data):
        """Write unblock actions to output file"""
        for user_id in users:
            user_data_item = user_data[user_id]
            
            for unblock in user_data_item['unblocks']:
                rkey = unblock.get('rkey', '')
                
                # Try to find the original block to get the blocked user
                original_block = None
                for block in user_data_item['blocks']:
                    if block.get('rkey', '') == rkey:
                        original_block = block
                        break
                
                if not original_block:
                    continue
                
                blocked_user = original_block.get('subject', '')
                
                if not blocked_user or blocked_user not in user_data:
                    continue
                
                # Find the most recent post from the unblocked user
                blocked_data = user_data[blocked_user]
                
                # Combine posts and replies for the unblocked user
                all_content = blocked_data['posts'] + blocked_data['replies']
                
                # Sort by created_at (most recent first)
                all_content.sort(key=lambda x: x.get('unix_epoch', 0), reverse=True)
                
                # Get the unblock timestamp
                unblock_time = unblock.get('unix_epoch', 0)
                
                # Find most recent post before the unblock
                # This is a bit awkward, because if the user was blocked,
                # their most recent post at time of unblock could not have been seen
                most_recent_post = None
                for content in all_content:
                    if content.get('unix_epoch', 0) <= unblock_time:
                        most_recent_post = content
                        break
                
                # If found, write the unblock action
                if most_recent_post:
                    content_msg = {'user_id': blocked_user}
                    if most_recent_post.get('text', ''):
                        content_msg['text'] = most_recent_post.get('text', '')
                    content_msg['unix_epoch'] = most_recent_post.get('unix_epoch', 0)
                    
                    action = default_actions()
                    action['unblock'] = True
                    action_msg = {
                        'user_id': user_id,
                        'actions': action,
                        'unix_epoch': unblock.get('unix_epoch', 0)
                    }
                    
                    action = [content_msg, action_msg]
                    file.write(json.dumps(action) + '\n')
    
    def write_post_update_actions(self, file, users, user_data):
        """Write post update actions to output file"""
        for user_id in users:
            user_data_item = user_data[user_id]
            
            for post_update in user_data_item['post_updates']:
                rkey = post_update.get('rkey', '')
                new_text = post_update.get('text', '')
                
                # Find the original post that was updated
                original_post = None
                for post in user_data_item['posts']:
                    if post.get('rkey', '') == rkey:
                        original_post = post
                        break
                
                # If not in posts, check replies
                if not original_post:
                    for reply in user_data_item['replies']:
                        if reply.get('rkey', '') == rkey:
                            original_post = reply
                            break
                
                if not original_post:
                    continue
                
                # Create the post update action
                content_msg = {'user_id': user_id}
                if original_post.get('text', ''):
                    content_msg['text'] = original_post.get('text', '')
                content_msg['unix_epoch'] = original_post.get('unix_epoch', 0)
                
                action = default_actions()
                action['post_update'] = True
                action_msg = {
                    'user_id': user_id,
                    'actions': action,
                    'unix_epoch': post_update.get('unix_epoch', 0)
                }
                if new_text:
                    action_msg['text'] = new_text
                
                action = [content_msg, action_msg]
                file.write(json.dumps(action) + '\n')
    
    def write_post_delete_actions(self, file, users, user_data):
        """Write post delete actions to output file"""
        for user_id in users:
            user_data_item = user_data[user_id]
            
            for post_delete in user_data_item['post_deletes']:
                rkey = post_delete.get('rkey', '')
                
                # Find the original post that was deleted
                original_post = None
                for post in user_data_item['posts']:
                    if post.get('rkey', '') == rkey:
                        original_post = post
                        break
                
                # If not in posts, check replies
                if not original_post:
                    for reply in user_data_item['replies']:
                        if reply.get('rkey', '') == rkey:
                            original_post = reply
                            break
                
                if not original_post:
                    continue
                
                # Create the post delete action
                content_msg = {'user_id': user_id}
                if original_post.get('text', ''):
                    content_msg['text'] = original_post.get('text', '')
                content_msg['unix_epoch'] = original_post.get('unix_epoch', 0)
                
                action = default_actions()
                action['post_delete'] = True
                action_msg = {
                    'user_id': user_id,
                    'actions': action,
                    'unix_epoch': post_delete.get('unix_epoch', 0)
                }
                
                action = [content_msg, action_msg]
                file.write(json.dumps(action) + '\n')
    
    def write_ignored_content(self, file, users, ignored_content):
        """Write ignored content to output file"""
        
        if len(ignored_content) > self.cap_ignored_messages * len(users):
            # Cap the number of ignored messages per user
            # Randomly select messages to keep if we have more than the cap
            ignored_content = np.random.choice(
                ignored_content,
                size=self.cap_ignored_messages * len(users),
                replace=False
            ).tolist()
        
        for content in ignored_content:
            if 'source_user' not in content:
                continue
                
            # Create the ignore action for a random user in the cluster
            random_user = users[np.random.randint(0, len(users))]
            
            content_msg = {'user_id': content.get('source_user')}
            if content.get('text', ''):
                content_msg['text'] = content.get('text', '')
            content_msg['unix_epoch'] = content.get('unix_epoch', 0)
            
            action = default_actions()
            action['ignore'] = True
            action_msg = {
                'user_id': random_user,
                'actions': action,
                'unix_epoch': content.get('unix_epoch', 0)
            }
            
            action = [content_msg, action_msg]
            file.write(json.dumps(action) + '\n')
    
    def write_post_quote_actions(self, file, users, user_data):
        """Write post quote actions to output file"""
        for user_id in users:
            user_data_item = user_data[user_id]
            
            for quote in user_data_item['quotes']:
                uri = quote.get('quoted', {}).get('uri', '')
                
                # Find the original post that was quoted
                post_content = self.find_content_by_uri(uri, user_data)
                
                if post_content and 'user_id' in post_content:
                    # Create the quote action
                    content_msg = {'user_id': post_content.get('user_id')}
                    if post_content.get('text', ''):
                        content_msg['text'] = post_content.get('text', '')
                    content_msg['unix_epoch'] = post_content.get('unix_epoch', 0)
                    
                    action = default_actions()
                    action['quote'] = True
                    action_msg = {
                        'user_id': user_id,
                        'actions': action,
                        'unix_epoch': quote.get('unix_epoch', 0)
                    }
                    if quote.get('text', False):
                        action_msg['text'] = quote.get('text', '')
                    
                    action = [content_msg, action_msg]
                    file.write(json.dumps(action) + '\n')
    
    def find_content_by_uri(self, uri, user_data):
        """Find content by URI"""
        # Extract user ID and post key from URI
        parts = uri.split('/')
        if len(parts) < 4:
            return None
            
        post_user_id = parts[2]
        post_key = parts[4] if len(parts) > 4 else None
        
        if not post_user_id or not post_key or post_user_id not in user_data:
            return None
            
        # Look in posts
        for post in user_data[post_user_id]['posts']:
            if post.get('rkey') == post_key:
                return {'text': post.get('text', ''), 'user_id': post_user_id, 'unix_epoch': post.get('unix_epoch', 0)}
        
        # Look in replies
        for reply in user_data[post_user_id]['replies']:
            if reply.get('rkey') == post_key:
                return {'text': reply.get('text', ''), 'user_id': post_user_id, 'unix_epoch': reply.get('unix_epoch', 0)}
        
        return None