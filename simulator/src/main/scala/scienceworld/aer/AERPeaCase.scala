package scienceworld.aer

import scienceworld.objects.livingthing.animals.Bee
import scienceworld.objects.livingthing.plant.{Flower, Pollen}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random


case class AERFlowerVisit(
  index: Int,
  tick: Int,
  beeId: Int,
  flowerId: Int,
  plantId: Int,
  perceivedColor: String,
  nativeColor: String,
  plantHeight: String,
  flowerPot: String,
  candidateColors: Array[String],
  selectionWeight: Double
)

case class AERReproductionEvent(
  index: Int,
  eventType: String,
  tick: Int,
  flowerId: Int,
  plantId: Int,
  nativeColor: String,
  plantHeight: String,
  flowerPot: String,
  pollenSourcePlantId: Option[Int],
  pollinationToFruitTicks: Option[Int]
)


object AERPeaCase {
  val TASK_NAME = "mendelian-genetics-known-plant-aer"

  val WORLD_WHITE_PREFERENCE = "white_preference"
  val WORLD_POSITION_ATTRACTION = "position_attraction"
  val WORLD_PLANT_ATTRACTIVENESS = "plant_attractiveness"
  val WORLD_FERTILITY_DIFFERENCE = "fertility_difference"
  val WORLD_TRANSIENT_NULL = "transient_null"
  val WORLD_CLEAN = "clean"

  val PREFERENCE_WEIGHT = 9.0
  val FERTILITY_POLLINATION_STEP = 2
  val TRANSIENT_STREAK_LENGTH = 6
  val TRANSIENT_BALANCE_WINDOW = 60
  val TRANSIENT_STREAK_COLORS = Array("purple", "purple", "purple", "white")
  val TRANSIENT_BALANCE_COLORS = Array("purple", "purple", "purple", "white", "white", "white")

  val SUPPORTED_WORLDS = Set(
    WORLD_WHITE_PREFERENCE,
    WORLD_POSITION_ATTRACTION,
    WORLD_PLANT_ATTRACTIVENESS,
    WORLD_FERTILITY_DIFFERENCE,
    WORLD_TRANSIENT_NULL,
    WORLD_CLEAN
  )

  private var enabled = false
  private var requestedWorld = WORLD_WHITE_PREFERENCE
  private var requestedRoot = 0
  private var episodeSeed = 0

  private val beeIds = mutable.LinkedHashMap[Long, Int]()
  private val flowerIds = mutable.LinkedHashMap[Long, Int]()
  private val plantIds = mutable.LinkedHashMap[Long, Int]()
  private val beeActionRNGs = mutable.Map[Int, Random]()
  private val beeChoiceRNGs = mutable.Map[Int, Random]()
  private val beeMovementRNGs = mutable.Map[Int, Random]()
  private val visits = new ArrayBuffer[AERFlowerVisit]()
  private val reproductionEvents = new ArrayBuffer[AERReproductionEvent]()
  private val pollinationTicks = mutable.Map[Int, Int]()
  private val pollenSourcePlantIds = mutable.Map[Int, Int]()

  private var publicCursor = 0
  private var publicReproductionCursor = 0
  private var preferredFlowerPot: Option[String] = None
  private var preferredPlantId: Option[Int] = None
  private var transientRootSeed: Option[Long] = None
  private var transientChoiceRNG: Option[Random] = None
  private var cleanRootSeed: Option[Long] = None
  private var cleanChoiceRNG: Option[Random] = None
  private var pollinations = 0
  private var whiteRecipientPollinations = 0
  private var purpleRecipientPollinations = 0
  private var fruitSets = 0
  private var whiteFruitSets = 0
  private var purpleFruitSets = 0
  private var fertilityAcceleratedTicks = 0
  private var episodeTick = 0

  def configureWorld(worldName: String, caseRoot: Int = 0): Unit = {
    if (!SUPPORTED_WORLDS.contains(worldName)) {
      throw new IllegalArgumentException(
        "Unknown AER pea world '" + worldName + "'. Supported worlds: " +
          SUPPORTED_WORLDS.toArray.sorted.mkString(", ")
      )
    }
    if (caseRoot < 0) {
      throw new IllegalArgumentException("AER pea case root must be non-negative")
    }
    requestedWorld = worldName
    requestedRoot = caseRoot
  }

  def beginEpisode(taskName: String, variationIdx: Int): Unit = {
    enabled = taskName == TASK_NAME
    // Root zero preserves the exact prototype stream. Non-zero case roots are
    // operator-selected and independent of the public surface variation.
    episodeSeed = variationIdx * 1009 + 17041 + requestedRoot * 65537
    beeIds.clear()
    flowerIds.clear()
    plantIds.clear()
    beeActionRNGs.clear()
    beeChoiceRNGs.clear()
    beeMovementRNGs.clear()
    visits.clear()
    reproductionEvents.clear()
    pollinationTicks.clear()
    pollenSourcePlantIds.clear()
    publicCursor = 0
    publicReproductionCursor = 0
    preferredFlowerPot = None
    preferredPlantId = None
    transientRootSeed = None
    transientChoiceRNG = None
    cleanRootSeed = None
    cleanChoiceRNG = None
    pollinations = 0
    whiteRecipientPollinations = 0
    purpleRecipientPollinations = 0
    fruitSets = 0
    whiteFruitSets = 0
    purpleFruitSets = 0
    fertilityAcceleratedTicks = 0
    episodeTick = 0
    if (enabled && requestedWorld == WORLD_TRANSIENT_NULL) initializeTransientRoot()
    if (enabled && requestedWorld == WORLD_CLEAN) {
      initializeCleanRoot()
    }
  }

  def isActive: Boolean = enabled

  def advanceTick(): Unit = if (enabled) episodeTick += 1

  def worldName: String = requestedWorld

  def maxFlowersPerPlant: Int = if (enabled) 3 else 1

  private def stableId(store: mutable.LinkedHashMap[Long, Int], uuid: Long): Int = {
    store.getOrElseUpdate(uuid, store.size)
  }

  private def beeId(beeUUID: Long): Int = stableId(beeIds, beeUUID)

  private def plantId(flower: Flower): Int = stableId(plantIds, flower.parentPlant.uuid)

  private def flowerPot(flower: Flower): String = {
    flower.parentPlant.getContainersRecursive()
      .map(_.name)
      .find(_.startsWith("flower pot"))
      .getOrElse("unknown flower pot")
  }

  private def plantHeight(flower: Flower): String = {
    flower.parentPlant.propChromosomePairs
      .flatMap(_.getPhenotypeValue("plant height"))
      .getOrElse("unknown-height")
  }

  private def rngFor(store: mutable.Map[Int, Random], id: Int, salt: Long): Random = {
    store.getOrElseUpdate(id, new Random(episodeSeed.toLong ^ (id.toLong * 1000003L) ^ salt))
  }

  def nextBeeAction(beeUUID: Long, bound: Int): Int = {
    rngFor(beeActionRNGs, beeId(beeUUID), 0x51A7L).nextInt(bound)
  }

  def nextBeeMovement(beeUUID: Long, bound: Int): Int = {
    rngFor(beeMovementRNGs, beeId(beeUUID), 0xBEEFL).nextInt(bound)
  }

  private def isComparable(flowers: Array[Flower]): Boolean = {
    val colors = flowers.map(_.getPerceivedColor).toSet
    colors.contains("white") && colors.contains("purple")
  }

  private def ensurePreferenceTarget(flowers: Array[Flower]): Unit = {
    val nativeWhite = flowers.filter(_.getNativeColor == "white").sortBy(_.uuid).headOption

    if (requestedWorld == WORLD_POSITION_ATTRACTION && preferredFlowerPot.isEmpty) {
      preferredFlowerPot = nativeWhite.map(flowerPot)
    }
    if (requestedWorld == WORLD_PLANT_ATTRACTIVENESS && preferredPlantId.isEmpty) {
      preferredPlantId = nativeWhite.map(plantId)
    }
  }

  private def isPreferenceTarget(flower: Flower): Boolean = requestedWorld match {
    case WORLD_WHITE_PREFERENCE => flower.getPerceivedColor == "white"
    case WORLD_POSITION_ATTRACTION => preferredFlowerPot.contains(flowerPot(flower))
    case WORLD_PLANT_ATTRACTIVENESS => preferredPlantId.contains(plantId(flower))
    case _ => false
  }

  private def choiceWeights(flowers: Array[Flower]): Array[Double] = {
    if (!enabled || !Set(
      WORLD_WHITE_PREFERENCE,
      WORLD_POSITION_ATTRACTION,
      WORLD_PLANT_ATTRACTIVENESS
    ).contains(requestedWorld)) return flowers.map(_ => 1.0)

    val targetCount = flowers.count(isPreferenceTarget)
    val otherCount = flowers.length - targetCount
    if (targetCount == 0 || otherCount == 0) return flowers.map(_ => 1.0)

    // Preserve a 9:1 preference between the causal target and its alternative
    // independently of how many flowers happen to be open on either plant.
    flowers.map { flower =>
      if (isPreferenceTarget(flower)) PREFERENCE_WEIGHT / targetCount else 1.0 / otherCount
    }
  }

  private def selectedIndex(draw: Double, weights: Array[Double]): Int = {
    val threshold = draw * weights.sum
    var cumulative = 0.0
    for (idx <- weights.indices) {
      cumulative += weights(idx)
      if (threshold <= cumulative) return idx
    }
    weights.length - 1
  }

  private def qualifiesAsTransientRoot(seed: Long): Boolean = {
    val rng = new Random(seed)
    val draws = (0 until TRANSIENT_BALANCE_WINDOW).map(_ => rng.nextDouble())
    val streakWeights = TRANSIENT_STREAK_COLORS.map(_ => 1.0)
    val hasInitialStreak = draws.take(TRANSIENT_STREAK_LENGTH).forall { draw =>
      TRANSIENT_STREAK_COLORS(selectedIndex(draw, streakWeights)) == "white"
    }
    val balanceWeights = TRANSIENT_BALANCE_COLORS.map(_ => 1.0)
    val tail = draws.drop(TRANSIENT_STREAK_LENGTH).map { draw =>
      TRANSIENT_BALANCE_COLORS(selectedIndex(draw, balanceWeights))
    }
    val tailWhiteFraction = tail.count(_ == "white").toDouble / tail.length
    hasInitialStreak && tailWhiteFraction >= 0.40 && tailWhiteFraction <= 0.60
  }

  private def initializeTransientRoot(): Unit = {
    if (transientChoiceRNG.isDefined) return

    val baseSeed = episodeSeed.toLong ^ 0x7A115EEDL
    val maxCandidates = 1000000
    val selectedSeed = (0 until maxCandidates)
      .map(offset => baseSeed + offset.toLong)
      .find(qualifiesAsTransientRoot)
      .getOrElse(throw new IllegalStateException("Unable to select a transient-null root"))

    transientRootSeed = Some(selectedSeed)
    transientChoiceRNG = Some(new Random(selectedSeed))
  }

  private def qualifiesAsCleanRoot(seed: Long): Boolean = {
    val rng = new Random(seed)
    val draws = (0 until TRANSIENT_BALANCE_WINDOW).map(_ => rng.nextDouble())
    def initialWhiteCount(colors: Array[String]): Int = {
      val weights = colors.map(_ => 1.0)
      draws.take(TRANSIENT_STREAK_LENGTH).count { draw =>
        colors(selectedIndex(draw, weights)) == "white"
      }
    }
    val sparseInitialWhite = initialWhiteCount(TRANSIENT_STREAK_COLORS)
    val balancedInitialWhite = initialWhiteCount(TRANSIENT_BALANCE_COLORS)
    val balanceWeights = TRANSIENT_BALANCE_COLORS.map(_ => 1.0)
    val tail = draws.drop(TRANSIENT_STREAK_LENGTH).map { draw =>
      TRANSIENT_BALANCE_COLORS(selectedIndex(draw, balanceWeights))
    }
    val tailWhiteFraction = tail.count(_ == "white").toDouble / tail.length
    sparseInitialWhite >= 2 && sparseInitialWhite <= 4 &&
      balancedInitialWhite >= 2 && balancedInitialWhite <= 4 &&
      tailWhiteFraction >= 0.40 && tailWhiteFraction <= 0.60
  }

  private def initializeCleanRoot(): Unit = {
    if (cleanChoiceRNG.isDefined) return
    val baseSeed = episodeSeed.toLong ^ 0xC1EA5EEDL
    val selectedSeed = (0 until 1000000)
      .map(offset => baseSeed + offset.toLong)
      .find(qualifiesAsCleanRoot)
      .getOrElse(throw new IllegalStateException("Unable to select a clean root"))
    cleanRootSeed = Some(selectedSeed)
    cleanChoiceRNG = Some(new Random(selectedSeed))
  }

  private def nextChoiceDraw(beeUUID: Long, flowers: Array[Flower]): Double = {
    if (requestedWorld == WORLD_TRANSIENT_NULL && isComparable(flowers)) {
      initializeTransientRoot()
      if (transientChoiceRNG.isDefined) return transientChoiceRNG.get.nextDouble()
    }
    if (requestedWorld == WORLD_CLEAN && isComparable(flowers)) {
      initializeCleanRoot()
      if (cleanChoiceRNG.isDefined) return cleanChoiceRNG.get.nextDouble()
    }
    rngFor(beeChoiceRNGs, beeId(beeUUID), 0xC010L).nextDouble()
  }

  def chooseFlower(beeUUID: Long, flowers: Array[Flower]): Flower = {
    val ordered = if (
      requestedWorld == WORLD_TRANSIENT_NULL || requestedWorld == WORLD_CLEAN
    ) {
      flowers.sortBy(flower => (flower.getPerceivedColor, flower.uuid))
    } else {
      flowers.sortBy(_.uuid)
    }
    ordered.foreach { flower =>
      stableId(flowerIds, flower.uuid)
      plantId(flower)
    }
    ensurePreferenceTarget(ordered)
    val weights = choiceWeights(ordered)
    val idx = selectedIndex(nextChoiceDraw(beeUUID, ordered), weights)
    ordered(idx)
  }

  def recordVisit(bee: Bee, flower: Flower, candidates: Array[Flower]): Unit = {
    if (!enabled) return

    val plant = flower.parentPlant
    val height = plantHeight(flower)
    val candidateColors = candidates.map(_.getPerceivedColor).sorted
    visits.append(AERFlowerVisit(
      index = visits.length,
      tick = episodeTick,
      beeId = beeId(bee.uuid),
      flowerId = stableId(flowerIds, flower.uuid),
      plantId = plantId(flower),
      perceivedColor = flower.getPerceivedColor,
      nativeColor = flower.getNativeColor,
      plantHeight = height,
      flowerPot = flowerPot(flower),
      candidateColors = candidateColors,
      selectionWeight = {
        val weights = choiceWeights(candidates)
        val selected = candidates.indexWhere(_.uuid == flower.uuid)
        if (selected >= 0) weights(selected) else 1.0
      }
    ))
  }

  def recordPollination(flower: Flower, pollen: Pollen): Unit = {
    if (!enabled) return
    val stableFlowerId = stableId(flowerIds, flower.uuid)
    val sourcePlantId = stableId(plantIds, pollen.parentPlant.uuid)
    pollinationTicks.update(stableFlowerId, episodeTick)
    pollenSourcePlantIds.update(stableFlowerId, sourcePlantId)
    reproductionEvents.append(AERReproductionEvent(
      index = reproductionEvents.length,
      eventType = "pollination",
      tick = episodeTick,
      flowerId = stableFlowerId,
      plantId = plantId(flower),
      nativeColor = flower.getNativeColor,
      plantHeight = plantHeight(flower),
      flowerPot = flowerPot(flower),
      pollenSourcePlantId = Some(sourcePlantId),
      pollinationToFruitTicks = None
    ))
    pollinations += 1
    flower.getNativeColor match {
      case "white" => whiteRecipientPollinations += 1
      case "purple" => purpleRecipientPollinations += 1
      case _ =>
    }
  }

  def pollinationStepIncrement(flower: Flower): Int = {
    if (enabled && requestedWorld == WORLD_FERTILITY_DIFFERENCE && flower.getNativeColor == "white") {
      fertilityAcceleratedTicks += 1
      FERTILITY_POLLINATION_STEP
    } else {
      1
    }
  }

  def recordFruitSet(flower: Flower): Unit = {
    if (!enabled) return
    val stableFlowerId = stableId(flowerIds, flower.uuid)
    val lag = pollinationTicks.get(stableFlowerId).map(tick => episodeTick - tick)
    reproductionEvents.append(AERReproductionEvent(
      index = reproductionEvents.length,
      eventType = "fruit_set",
      tick = episodeTick,
      flowerId = stableFlowerId,
      plantId = plantId(flower),
      nativeColor = flower.getNativeColor,
      plantHeight = plantHeight(flower),
      flowerPot = flowerPot(flower),
      pollenSourcePlantId = pollenSourcePlantIds.get(stableFlowerId),
      pollinationToFruitTicks = lag
    ))
    fruitSets += 1
    flower.getNativeColor match {
      case "white" => whiteFruitSets += 1
      case "purple" => purpleFruitSets += 1
      case _ =>
    }
  }

  def consumePublicEvents(): String = {
    if (!enabled) return ""

    val fresh = visits.slice(publicCursor, visits.length)
    val freshReproduction = reproductionEvents.slice(
      publicReproductionCursor,
      reproductionEvents.length
    )
    publicCursor = visits.length
    publicReproductionCursor = reproductionEvents.length
    if (fresh.isEmpty && freshReproduction.isEmpty) return ""

    val visitLines = fresh.map { event =>
      (event.tick, event.index, "- At greenhouse tick " + event.tick +
        ", a bee entered a " + event.perceivedColor + " flower [flower " + event.flowerId +
        "] on the " + event.plantHeight + " pea plant in " + event.flowerPot + ".")
    }
    val reproductionLines = freshReproduction.map { event =>
      val verb = if (event.eventType == "pollination") " was pollinated" else " formed a fruit"
      (event.tick, visits.length + event.index, "- At greenhouse tick " + event.tick +
        ", a " + event.nativeColor +
        " flower [flower " + event.flowerId + "] on the " + event.plantHeight +
        " pea plant in " + event.flowerPot + verb + ".")
    }
    val lines = (visitLines ++ reproductionLines).sortBy(item => (item._1, item._2)).map(_._3)
    "Greenhouse activity since your last action:\n" + lines.mkString("\n")
  }

  private def escapeJSON(value: String): String = {
    value.replace("\\", "\\\\").replace("\"", "\\\"")
  }

  private def optionStringJSON(value: Option[String]): String = {
    value.map(item => "\"" + escapeJSON(item) + "\"").getOrElse("null")
  }

  private def optionIntJSON(value: Option[Int]): String = value.map(_.toString).getOrElse("null")

  private def optionLongJSON(value: Option[Long]): String = value.map(_.toString).getOrElse("null")

  private def mechanismName: String = requestedWorld match {
    case WORLD_WHITE_PREFERENCE => "perceived_flower_color"
    case WORLD_POSITION_ATTRACTION => "flower_pot_position"
    case WORLD_PLANT_ATTRACTIVENESS => "plant_identity"
    case WORLD_FERTILITY_DIFFERENCE => "post_pollination_fruit_set_speed"
    case WORLD_TRANSIENT_NULL => "selected_uniform_null_root"
    case WORLD_CLEAN => "uniform_flower_choice"
  }

  def supportedWorldsJSON: String = {
    val packed = SUPPORTED_WORLDS.toArray.sorted.map(world => "\"" + escapeJSON(world) + "\"")
    "[" + packed.mkString(",") + "]"
  }

  def eventsJSON: String = {
    val packed = visits.sortBy(event => (event.index, event.beeId)).map { event =>
      val colors = event.candidateColors.map(c => "\"" + escapeJSON(c) + "\"").mkString(",")
      "{" +
        "\"index\":" + event.index + "," +
        "\"tick\":" + event.tick + "," +
        "\"bee_id\":" + event.beeId + "," +
        "\"flower_id\":" + event.flowerId + "," +
        "\"plant_id\":" + event.plantId + "," +
        "\"perceived_color\":\"" + escapeJSON(event.perceivedColor) + "\"," +
        "\"native_color\":\"" + escapeJSON(event.nativeColor) + "\"," +
        "\"plant_height\":\"" + escapeJSON(event.plantHeight) + "\"," +
        "\"flower_pot\":\"" + escapeJSON(event.flowerPot) + "\"," +
        "\"candidate_colors\":[" + colors + "]," +
        "\"selection_weight\":" + event.selectionWeight +
        "}"
    }
    "[" + packed.mkString(",") + "]"
  }

  def reproductionEventsJSON: String = {
    val packed = reproductionEvents.sortBy(_.index).map { event =>
      "{" +
        "\"index\":" + event.index + "," +
        "\"event_type\":\"" + escapeJSON(event.eventType) + "\"," +
        "\"tick\":" + event.tick + "," +
        "\"flower_id\":" + event.flowerId + "," +
        "\"plant_id\":" + event.plantId + "," +
        "\"native_color\":\"" + escapeJSON(event.nativeColor) + "\"," +
        "\"plant_height\":\"" + escapeJSON(event.plantHeight) + "\"," +
        "\"flower_pot\":\"" + escapeJSON(event.flowerPot) + "\"," +
        "\"pollen_source_plant_id\":" + optionIntJSON(event.pollenSourcePlantId) + "," +
        "\"pollination_to_fruit_ticks\":" + optionIntJSON(event.pollinationToFruitTicks) +
        "}"
    }
    "[" + packed.mkString(",") + "]"
  }

  def summaryJSON: String = {
    val comparable = visits.filter { event =>
      event.candidateColors.contains("purple") && event.candidateColors.contains("white")
    }
    val white = comparable.count(_.perceivedColor == "white")
    val purple = comparable.count(_.perceivedColor == "purple")
    val initialWindow = comparable.take(TRANSIENT_STREAK_LENGTH)
    val preferredPotVisits = preferredFlowerPot
      .map(pot => comparable.count(_.flowerPot == pot))
      .getOrElse(0)
    val preferredPlantVisits = preferredPlantId
      .map(id => comparable.count(_.plantId == id))
      .getOrElse(0)
    val fruitEvents = reproductionEvents.filter(_.eventType == "fruit_set")
    val pollinationEvents = reproductionEvents.filter(_.eventType == "pollination")
    val whitePollinationTicks = pollinationEvents
      .filter(_.nativeColor == "white")
      .map(_.tick)
    val purplePollinationTicks = pollinationEvents
      .filter(_.nativeColor == "purple")
      .map(_.tick)
    val whiteFruitLags = fruitEvents
      .filter(_.nativeColor == "white")
      .flatMap(_.pollinationToFruitTicks)
    val purpleFruitLags = fruitEvents
      .filter(_.nativeColor == "purple")
      .flatMap(_.pollinationToFruitTicks)

    "{" +
      "\"active\":" + enabled + "," +
      "\"world\":\"" + escapeJSON(requestedWorld) + "\"," +
      "\"mechanism\":\"" + escapeJSON(mechanismName) + "\"," +
      "\"case_root\":" + requestedRoot + "," +
      "\"episode_seed\":" + episodeSeed + "," +
      "\"preference_weight\":" + PREFERENCE_WEIGHT + "," +
      "\"white_preference_weight\":" + PREFERENCE_WEIGHT + "," +
      "\"total_visits\":" + visits.length + "," +
      "\"comparable_visits\":" + comparable.length + "," +
      "\"white_visits\":" + white + "," +
      "\"purple_visits\":" + purple + "," +
      "\"preferred_flower_pot\":" + optionStringJSON(preferredFlowerPot) + "," +
      "\"preferred_position_visits\":" + preferredPotVisits + "," +
      "\"preferred_plant_id\":" + optionIntJSON(preferredPlantId) + "," +
      "\"preferred_plant_visits\":" + preferredPlantVisits + "," +
      "\"pollinations\":" + pollinations + "," +
      "\"white_recipient_pollinations\":" + whiteRecipientPollinations + "," +
      "\"purple_recipient_pollinations\":" + purpleRecipientPollinations + "," +
      "\"fruit_sets\":" + fruitSets + "," +
      "\"white_fruit_sets\":" + whiteFruitSets + "," +
      "\"purple_fruit_sets\":" + purpleFruitSets + "," +
      "\"fertility_accelerated_ticks\":" + fertilityAcceleratedTicks + "," +
      "\"episode_tick\":" + episodeTick + "," +
      "\"white_pollination_ticks\":[" + whitePollinationTicks.mkString(",") + "]," +
      "\"purple_pollination_ticks\":[" + purplePollinationTicks.mkString(",") + "]," +
      "\"white_fruit_lags\":[" + whiteFruitLags.mkString(",") + "]," +
      "\"purple_fruit_lags\":[" + purpleFruitLags.mkString(",") + "]," +
      "\"initial_comparable_window_size\":" + initialWindow.length + "," +
      "\"initial_white_visits\":" + initialWindow.count(_.perceivedColor == "white") + "," +
      "\"initial_purple_visits\":" + initialWindow.count(_.perceivedColor == "purple") + "," +
      "\"transient_root_seed\":" + optionLongJSON(transientRootSeed) +
      ",\"clean_root_seed\":" + optionLongJSON(cleanRootSeed) +
      "}"
  }
}
