import numpy as np
from tqdm import tqdm
from utils import logger, cosine_similarity

class ContentAnalyzer:
    def __init__(self, similarity_threshold=0.7):
        self.similarity_threshold = similarity_threshold
    
    def find_seen_content(self, cluster_id, cluster_users, user_data):
        """Find content users in a cluster have likely seen but not interacted with"""
        users = cluster_users[cluster_id]
        
        # Collect all content that users in cluster have interacted with
        interacted_content = []
        followed_users = set()
        
        for user_id in tqdm(users, desc=f"Collect interacted content"):
            user_data_item = user_data[user_id]
            
            # Add liked content
            for like in user_data_item['likes']:
                uri = like.get('uri', '')
                if uri:
                    interacted_content.append({
                        'uri': uri,
                        'cid': like.get('cid', ''),
                        'interaction_type': 'like',
                        'interacting_user': user_id
                    })
            
            # Add reposted content
            for repost in user_data_item['reposts']:
                uri = repost.get('uri', '')
                if uri:
                    interacted_content.append({
                        'uri': uri,
                        'cid': repost.get('cid', ''),
                        'interaction_type': 'repost',
                        'interacting_user': user_id
                    })
            
            # Add followed users
            for follow in user_data_item['follows']:
                followed_user = follow.get('subject', '')
                if followed_user:
                    followed_users.add(followed_user)
        
        # Collect posts from followed users
        followed_content = []

        # Build follower relationships linearly
        follower_map = {user: [] for user in followed_users}
        
        # Process each user in the cluster and record their follow relationships
        for user_id in tqdm(users, desc="Building follower relationships"):
            # For each follow action by this user
            for follow in user_data[user_id]['follows']:
                followed_user = follow.get('subject', '')
                # If this followed user is in our map, add this user as their follower
                if followed_user in follower_map:
                    follower_map[followed_user].append(user_id)

        # Process only valid followed users (ones that exist in user_data)
        valid_followed_users = [user for user in followed_users if user in user_data]

        # Process in batches to improve performance
        batch_size = 1000
        for i in tqdm(range(0, len(valid_followed_users), batch_size), desc="Process followed users in batches"):
            batch_users = valid_followed_users[i:i+batch_size]
            for followed_user in tqdm(batch_users, desc=f"Collect followed content (batch {i//batch_size+1})"):
                # Get followers once per user
                followers = follower_map[followed_user]
                # Skip users with no followers in our cluster
                if not followers:
                    continue
                    
                # Process all posts for this user
                for post in user_data[followed_user]['posts']:
                    followed_content.append({
                        'uri': post.get('uri', ''),
                        'cid': post.get('cid', ''),
                        'text': post.get('text', ''),
                        'embedding': post.get('embedding', None),
                        'source_user': followed_user,
                        'followed_by': followers  # Use pre-computed followers list
                    })
        
        # Combine interacted and followed content
        all_seen_content = interacted_content + followed_content
        
        # Get embeddings for content that has text
        content_with_embeddings = []
        
        for content in tqdm(all_seen_content, desc=f"Get embeddings for seen content"):
            # Skip content without URI
            if not content.get('uri', ''):
                continue
            
            # For interacted content, need to find the original post to get its text
            if 'interaction_type' in content:
                # Extract user ID and post key from URI
                uri = content['uri']
                parts = uri.split('/')
                if len(parts) >= 4:
                    post_user_id = parts[2]
                    post_key = parts[4] if len(parts) > 4 else None
                    
                    # Find the original post
                    if post_user_id in user_data and post_key:
                        found = False
                        # Look in posts
                        for post in user_data[post_user_id]['posts']:
                            if post.get('rkey') == post_key:
                                content['text'] = post.get('text', '')
                                content['embedding'] = post.get('embedding', None)
                                found = True
                                break
                        
                        # Look in replies if not found in posts
                        if not found:
                            for reply in user_data[post_user_id]['replies']:
                                if reply.get('rkey') == post_key:
                                    content['text'] = reply.get('text', '')
                                    content['embedding'] = reply.get('embedding', None)
                                    break
            
            # Add to content with embeddings if it has an embedding
            if 'embedding' in content and content['embedding'] is not None:
                content_with_embeddings.append(content)
            elif 'text' in content and content['text']:
                # Generate embedding if there's text but no embedding
                content['embedding'] = None  # Would be set from the embedding manager
                content_with_embeddings.append(content)
        
        return content_with_embeddings
    
    def find_ignored_content(self, cluster_users, cluster_id, weighted_centers, user_data):
        """Find content likely seen but ignored by users in the cluster"""
        users = cluster_users[cluster_id]
        
        logger.info(f"Finding ignored content for cluster {cluster_id} with {len(users)} users")
        
        # Collect all URIs that users have interacted with
        interacted_uris = set()
        for user_id in tqdm(users, desc=f"Collect interacted URIs"):
            user_data_item = user_data[user_id]
            
            # Add liked URIs
            for like in user_data_item['likes']:
                interacted_uris.add(like.get('uri', ''))
            
            # Add reposted URIs
            for repost in user_data_item['reposts']:
                interacted_uris.add(repost.get('uri', ''))
        
        # Collect potential ignored content from all users (not just in this cluster)
        # This simulates the content that might have appeared in their feeds
        ignored_candidates = []
        
        for user_id, user_data_item in tqdm(user_data.items(), desc="Collect ignored candidates"):
            # Check posts
            for post in user_data_item['posts']:
                uri = post.get('uri', '')
                if uri and uri not in interacted_uris and 'embedding' in post:
                    ignored_candidates.append({
                        'uri': uri,
                        'text': post.get('text', ''),
                        'embedding': post['embedding'],
                        'source_user': user_id
                    })
            
            # Check replies
            for reply in user_data_item['replies']:
                uri = reply.get('uri', '')
                if uri and uri not in interacted_uris and 'embedding' in reply:
                    ignored_candidates.append({
                        'uri': uri,
                        'text': reply.get('text', ''),
                        'embedding': reply['embedding'],
                        'source_user': user_id
                    })
        
        # Find ignored content near weighted centers
        ignored_content = []
        
        for candidate in tqdm(ignored_candidates, desc="Find ignored content"):
            if 'embedding' not in candidate:
                continue
                
            # Check distance to each center
            for center in weighted_centers:
                similarity = cosine_similarity(candidate['embedding'], center)
                
                if similarity > self.similarity_threshold:
                    candidate['similarity'] = similarity
                    ignored_content.append(candidate)
                    break  # Once we find a match, no need to check other centers
        
        logger.info(f"Found {len(ignored_content)} ignored content items for cluster {cluster_id}")
        return ignored_content