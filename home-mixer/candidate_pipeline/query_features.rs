use serde::{Deserialize, Serialize};

/// User-specific features for personalization and filtering.
/// These features are hydrated from various sources (Strato, SGS, etc.)
/// and used throughout the candidate pipeline for filtering and scoring.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct UserFeatures {
    /// Keywords the user has muted - used for content filtering
    pub muted_keywords: Vec<String>,
    
    /// User IDs the user has blocked - posts from these users are filtered out
    pub blocked_user_ids: Vec<i64>,
    
    /// User IDs the user has muted - posts from these users are filtered out
    pub muted_user_ids: Vec<i64>,
    
    /// User IDs the user follows - used for in-network candidate sourcing
    pub followed_user_ids: Vec<i64>,
    
    /// User IDs the user subscribes to - used for subscription-based filtering
    pub subscribed_user_ids: Vec<i64>,
}
