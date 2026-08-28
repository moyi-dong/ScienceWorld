package scienceworld.aer

import scienceworld.objects.livingthing.animals.Bee
import scienceworld.objects.containers.{FlowerPot, SelfWateringFlowerPot}
import scienceworld.objects.livingthing.plant.{Flower, Plant, Pollen, Soil}
import scienceworld.objects.substance.Water
import scienceworld.struct.EnvObject

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
  pollenSourceHeight: Option[String],
  pollinationToFruitTicks: Option[Int]
)

case class AERCrossAttempt(
  attemptId: Int,
  flowerId: Int,
  recipientPlantId: Int,
  recipientPot: String,
  intendedPollenPlantId: Int,
  intendedPollenPot: String,
  actualPollenPlantId: Int,
  emasculated: Boolean,
  bagged: Boolean,
  contaminationOccurred: Boolean,
  startedTick: Int,
  var status: String,
  var resolvedTick: Option[Int],
  var podId: Option[Int],
  var seedIds: Array[String]
)


object AERPeaCase {
  val TASK_NAME = "mendelian-genetics-known-plant-aer"

  val WORLD_WHITE_PREFERENCE = "white_preference"
  val WORLD_POSITION_ATTRACTION = "position_attraction"
  val WORLD_PLANT_ATTRACTIVENESS = "plant_attractiveness"
  val WORLD_FERTILITY_DIFFERENCE = "fertility_difference"
  val WORLD_CROSS_DIRECTION_DELAY = "cross_direction_delay"
  val WORLD_TRANSIENT_NULL = "transient_null"
  val WORLD_CLEAN = "clean"

  val PREFERENCE_WEIGHT = 9.0
  val FERTILITY_POLLINATION_STEP = 2
  val TRANSIENT_STREAK_LENGTH = 6
  val TRANSIENT_BALANCE_WINDOW = 60
  val TRANSIENT_STREAK_COLORS = Array("purple", "purple", "purple", "white")
  val TRANSIENT_BALANCE_COLORS = Array("purple", "purple", "purple", "white", "white", "white")
  val LEGACY_FLOWER_POT_COUNT = 6
  val V04_FLOWER_POT_COUNT = 20

  val SUPPORTED_WORLDS = Set(
    WORLD_WHITE_PREFERENCE,
    WORLD_POSITION_ATTRACTION,
    WORLD_PLANT_ATTRACTIVENESS,
    WORLD_FERTILITY_DIFFERENCE,
    WORLD_CROSS_DIRECTION_DELAY,
    WORLD_TRANSIENT_NULL,
    WORLD_CLEAN
  )

  private var enabled = false
  private var requestedWorld = WORLD_WHITE_PREFERENCE
  private var requestedRoot = 0
  private var requestedPreferenceWeight = PREFERENCE_WEIGHT
  // The three integer slots are retained in the Java/Python ABI, but v0.4 gives
  // them new meanings: soil lot, fruit-set success, and parentage contamination.
  private var requestedSoilNutrientNoise = 0
  private var requestedFruitSetNoise = 0
  private var requestedContaminationNoise = 0
  private var requestedProtocolV04 = false
  private var episodeSeed = 0

  private val beeIds = mutable.LinkedHashMap[Long, Int]()
  private val flowerIds = mutable.LinkedHashMap[Long, Int]()
  private val plantIds = mutable.LinkedHashMap[Long, Int]()
  private val beeActionRNGs = mutable.Map[Int, Random]()
  private val beeChoiceRNGs = mutable.Map[Int, Random]()
  private val beeMovementRNGs = mutable.Map[Int, Random]()
  private val growthRNGs = mutable.Map[(Int, String), Random]()
  private val fruitSetRNGs = mutable.Map[Int, Random]()
  private val contaminationRNGs = mutable.Map[Int, Random]()
  private val soilLotIndexes = mutable.Map[Int, Int]()
  private val visits = new ArrayBuffer[AERFlowerVisit]()
  private val reproductionEvents = new ArrayBuffer[AERReproductionEvent]()
  private val pollinationTicks = mutable.Map[Int, Int]()
  private val pollenSourcePlantIds = mutable.Map[Int, Int]()
  private val pollenSourceHeights = mutable.Map[Int, String]()
  private val seedMaternalPlantIds = mutable.Map[Long, Int]()
  private val seedPodIds = mutable.Map[Long, Int]()
  private val seedIntendedPollenPlantIds = mutable.Map[Long, Int]()
  private val flowerAttemptIds = mutable.Map[Int, Int]()
  private val crossAttempts = new ArrayBuffer[AERCrossAttempt]()
  private val nextFlowerTicks = mutable.Map[Int, Int]()
  private var nextPodId = 0

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
    configureWorld(worldName, caseRoot, PREFERENCE_WEIGHT)
  }

  def configureWorld(worldName: String, caseRoot: Int, preferenceWeight: Double): Unit = {
    configureWorld(worldName, caseRoot, preferenceWeight, 0, 0, 0)
  }

  def configureWorld(
    worldName: String,
    caseRoot: Int,
    preferenceWeight: Double,
    soilNutrientNoise: Int,
    fruitSetNoise: Int,
    contaminationNoise: Int
  ): Unit = {
    if (!SUPPORTED_WORLDS.contains(worldName)) {
      throw new IllegalArgumentException(
        "Unknown AER pea world '" + worldName + "'. Supported worlds: " +
          SUPPORTED_WORLDS.toArray.sorted.mkString(", ")
      )
    }
    if (caseRoot < 0) {
      throw new IllegalArgumentException("AER pea case root must be non-negative")
    }
    if (preferenceWeight.isNaN || preferenceWeight.isInfinity || preferenceWeight < 1.0) {
      throw new IllegalArgumentException(
        "AER pea preference weight must be finite and at least 1.0"
      )
    }
    val noiseLevels = Array(soilNutrientNoise, fruitSetNoise, contaminationNoise)
    if (noiseLevels.exists(level => level < 0 || level > 3)) {
      throw new IllegalArgumentException("AER pea noise levels must be integers from 0 to 3")
    }
    requestedWorld = worldName
    requestedRoot = caseRoot
    requestedPreferenceWeight = preferenceWeight
    requestedSoilNutrientNoise = soilNutrientNoise
    requestedFruitSetNoise = fruitSetNoise
    requestedContaminationNoise = contaminationNoise
    requestedProtocolV04 = false
  }

  def configureWorldV04(
    worldName: String,
    caseRoot: Int,
    preferenceWeight: Double,
    soilNutrientNoise: Int,
    fruitSetNoise: Int,
    contaminationNoise: Int
  ): Unit = {
    configureWorld(
      worldName, caseRoot, preferenceWeight,
      soilNutrientNoise, fruitSetNoise, contaminationNoise
    )
    requestedProtocolV04 = true
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
    growthRNGs.clear()
    fruitSetRNGs.clear()
    contaminationRNGs.clear()
    soilLotIndexes.clear()
    visits.clear()
    reproductionEvents.clear()
    pollinationTicks.clear()
    pollenSourcePlantIds.clear()
    pollenSourceHeights.clear()
    seedMaternalPlantIds.clear()
    seedPodIds.clear()
    seedIntendedPollenPlantIds.clear()
    flowerAttemptIds.clear()
    crossAttempts.clear()
    nextFlowerTicks.clear()
    nextPodId = 0
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

  def isV04: Boolean = enabled && requestedProtocolV04

  def flowerPotCountForTask(isAERTask: Boolean): Int = {
    if (isAERTask && isV04) V04_FLOWER_POT_COUNT else LEGACY_FLOWER_POT_COUNT
  }

  def advanceTick(): Unit = if (enabled) episodeTick += 1

  def worldName: String = requestedWorld

  private def noiseStallProbability(level: Int): Double = level match {
    case 1 => 0.10
    case 2 => 0.25
    case 3 => 0.40
    case _ => 0.0
  }

  def growthStepIncrement(plant: Plant, stageName: String): Int = {
    if (enabled && !requestedProtocolV04) {
      if (requestedSoilNutrientNoise == 0) return 1
      val id = stableId(plantIds, plant.uuid)
      val key = (id, stageName)
      val rng = growthRNGs.getOrElseUpdate(
        key,
        new Random(episodeSeed.toLong ^ (id.toLong * 1000003L) ^
          (0x6A09E667L ^ stageName.hashCode.toLong))
      )
      return if (rng.nextDouble() < noiseStallProbability(requestedSoilNutrientNoise)) 0 else 1
    }
    if (!enabled || requestedSoilNutrientNoise == 0 || stageName == "seed") return 1
    val id = stableId(plantIds, plant.uuid)
    val key = (id, stageName)
    val salt = 0x6A09E667L ^ stageName.hashCode.toLong
    val rng = growthRNGs.getOrElseUpdate(
      key,
      new Random(episodeSeed.toLong ^ (id.toLong * 1000003L) ^ salt)
    )
    val multiplier = soilLotMultiplier(plant)
    if (rng.nextDouble() < 1.0 / multiplier) 1 else 0
  }

  def maxFlowersPerPlant(plant: Plant): Int = {
    if (!enabled || requestedProtocolV04) return 1
    val id = stableId(plantIds, plant.uuid)
    val rng = new Random(episodeSeed.toLong ^ (id.toLong * 1000003L) ^ 0xF10A3E5L)
    requestedFruitSetNoise match {
      case 1 => if (rng.nextDouble() < 0.25) 2 else 3
      case 2 => 1 + rng.nextInt(3)
      case 3 => if (rng.nextDouble() < 0.70) 1 else 2
      case _ => 3
    }
  }

  private def soilMultipliers(level: Int): Array[Double] = level match {
    case 1 => Array(1.00, 1.03, 1.05)
    case 2 => Array(1.00, 1.08, 1.15)
    case 3 => Array(1.00, 1.15, 1.30)
    case _ => Array(1.00)
  }

  private def soilLotIndex(plant: Plant): Int = {
    val id = stableId(plantIds, plant.uuid)
    soilLotIndexes.getOrElseUpdate(id, {
      val lots = soilMultipliers(requestedSoilNutrientNoise)
      val rng = new Random(episodeSeed.toLong ^ (id.toLong * 1000003L) ^ 0x5011107L)
      rng.nextInt(lots.length)
    })
  }

  def soilLotMultiplier(plant: Plant): Double = {
    val lots = soilMultipliers(requestedSoilNutrientNoise)
    lots(soilLotIndex(plant))
  }

  def soilLotName(plant: Plant): String = "soil-lot-" + soilLotIndex(plant)

  def canCreateFlower(plant: Plant): Boolean = {
    if (!requestedProtocolV04) return true
    val id = stableId(plantIds, plant.uuid)
    episodeTick >= nextFlowerTicks.getOrElse(id, 0)
  }

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

  private def plantPot(plant: Plant): String = {
    plant.getContainersRecursive()
      .map(_.name)
      .find(_.startsWith("flower pot"))
      .getOrElse("unknown flower pot")
  }

  private def plantHeight(flower: Flower): String = {
    plantHeight(flower.parentPlant)
  }

  private def plantHeight(plant: Plant): String = {
    plant.propChromosomePairs
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

    // Preserve the configured group-level preference between the causal target and its alternative
    // independently of how many flowers happen to be open on either plant.
    flowers.map { flower =>
      if (isPreferenceTarget(flower)) requestedPreferenceWeight / targetCount else 1.0 / otherCount
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
    pollenSourceHeights.update(stableFlowerId, plantHeight(pollen.parentPlant))
    if (!flowerAttemptIds.contains(stableFlowerId)) {
      val attemptId = crossAttempts.length
      flowerAttemptIds.update(stableFlowerId, attemptId)
      crossAttempts.append(AERCrossAttempt(
        attemptId = attemptId,
        flowerId = stableFlowerId,
        recipientPlantId = plantId(flower),
        recipientPot = flowerPot(flower),
        intendedPollenPlantId = sourcePlantId,
        intendedPollenPot = plantPot(pollen.parentPlant),
        actualPollenPlantId = sourcePlantId,
        emasculated = false,
        bagged = false,
        contaminationOccurred = false,
        startedTick = episodeTick,
        status = "pending",
        resolvedTick = None,
        podId = None,
        seedIds = Array.empty[String]
      ))
    }
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
      pollenSourceHeight = Some(plantHeight(pollen.parentPlant)),
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
    if (!enabled) return 1
    val stableFlowerId = stableId(flowerIds, flower.uuid)
    if (!requestedProtocolV04) {
      val rng = fruitSetRNGs.getOrElseUpdate(
        stableFlowerId,
        new Random(episodeSeed.toLong ^ (stableFlowerId.toLong * 1000003L) ^ 0xF81771E5L)
      )
      if (rng.nextDouble() < noiseStallProbability(requestedContaminationNoise)) return 0
    }
    if (requestedWorld == WORLD_FERTILITY_DIFFERENCE && flower.getNativeColor == "white") {
      fertilityAcceleratedTicks += 1
      FERTILITY_POLLINATION_STEP
    } else if (
      requestedWorld == WORLD_CROSS_DIRECTION_DELAY &&
      plantHeight(flower) == "short" &&
      pollenSourceHeights.get(stableFlowerId).contains("tall")
    ) {
      FERTILITY_POLLINATION_STEP
    } else {
      1
    }
  }

  private def fruitSetProbability: Double = requestedFruitSetNoise match {
    case 1 => 0.95
    case 2 => 0.85
    case 3 => 0.70
    case _ => 1.00
  }

  private def contaminationProbability(emasculated: Boolean, bagged: Boolean): Double = {
    if (emasculated && bagged) return 0.0
    val base = requestedContaminationNoise match {
      case 1 => 0.01
      case 2 => 0.05
      case 3 => 0.10
      case _ => 0.0
    }
    if (emasculated && !bagged) base * 0.5 else base
  }

  def shouldSetPod(flower: Flower): Boolean = {
    if (!enabled) return true
    val stableFlowerId = stableId(flowerIds, flower.uuid)
    val rng = fruitSetRNGs.getOrElseUpdate(
      stableFlowerId,
      new Random(episodeSeed.toLong ^ (stableFlowerId.toLong * 1000003L) ^ 0xF2A175E7L)
    )
    rng.nextDouble() < fruitSetProbability
  }

  def recordAborted(flower: Flower): Unit = {
    if (!enabled) return
    val stableFlowerId = stableId(flowerIds, flower.uuid)
    flowerAttemptIds.get(stableFlowerId).foreach { id =>
      val attempt = crossAttempts(id)
      attempt.status = "aborted"
      attempt.resolvedTick = Some(episodeTick)
    }
    nextFlowerTicks.update(plantId(flower), episodeTick + 2)
    reproductionEvents.append(AERReproductionEvent(
      index = reproductionEvents.length,
      eventType = "aborted",
      tick = episodeTick,
      flowerId = stableFlowerId,
      plantId = plantId(flower),
      nativeColor = flower.getNativeColor,
      plantHeight = plantHeight(flower),
      flowerPot = flowerPot(flower),
      pollenSourcePlantId = pollenSourcePlantIds.get(stableFlowerId),
      pollenSourceHeight = pollenSourceHeights.get(stableFlowerId),
      pollinationToFruitTicks = pollinationTicks.get(stableFlowerId).map(episodeTick - _)
    ))
  }

  def recordPodSet(flower: Flower, seeds: Array[EnvObject]): Unit = {
    if (!enabled) return
    val stableFlowerId = stableId(flowerIds, flower.uuid)
    val lag = pollinationTicks.get(stableFlowerId).map(tick => episodeTick - tick)
    val podId = nextPodId
    nextPodId += 1
    val publicSeedIds = seeds.collect { case seed: Plant =>
      val seedId = "seed-" + stableId(plantIds, seed.uuid)
      seedMaternalPlantIds.update(seed.uuid, plantId(flower))
      seedPodIds.update(seed.uuid, podId)
      flowerAttemptIds.get(stableFlowerId).foreach { attemptId =>
        seedIntendedPollenPlantIds.update(seed.uuid, crossAttempts(attemptId).intendedPollenPlantId)
      }
      seedId
    }
    flowerAttemptIds.get(stableFlowerId).foreach { id =>
      val attempt = crossAttempts(id)
      attempt.status = "pod_set"
      attempt.resolvedTick = Some(episodeTick)
      attempt.podId = Some(podId)
      attempt.seedIds = publicSeedIds
    }
    nextFlowerTicks.update(plantId(flower), episodeTick + 2)
    reproductionEvents.append(AERReproductionEvent(
      index = reproductionEvents.length,
      eventType = "pod_set",
      tick = episodeTick,
      flowerId = stableFlowerId,
      plantId = plantId(flower),
      nativeColor = flower.getNativeColor,
      plantHeight = plantHeight(flower),
      flowerPot = flowerPot(flower),
      pollenSourcePlantId = pollenSourcePlantIds.get(stableFlowerId),
      pollenSourceHeight = pollenSourceHeights.get(stableFlowerId),
      pollinationToFruitTicks = lag
    ))
    fruitSets += 1
    flower.getNativeColor match {
      case "white" => whiteFruitSets += 1
      case "purple" => purpleFruitSets += 1
      case _ =>
    }
  }

  def recordFruitSet(flower: Flower, fruit: EnvObject): Unit = {
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
      pollenSourceHeight = pollenSourceHeights.get(stableFlowerId),
      pollinationToFruitTicks = lag
    ))
    fruit match {
      case seed: Plant => seedMaternalPlantIds.update(seed.uuid, plantId(flower))
      case _ =>
    }
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
      val verb = event.eventType match {
        case "pollination" => " was pollinated"
        case "pod_set" => " formed a pod with four seeds"
        case "fruit_set" => " formed a fruit"
        case "aborted" => " aborted without forming a pod"
        case _ => " changed reproductive state"
      }
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
    case WORLD_CROSS_DIRECTION_DELAY => "cross_direction_fruit_set_delay"
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
        "\"pollen_source_height\":" + optionStringJSON(event.pollenSourceHeight) + "," +
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
    val fruitEvents = reproductionEvents.filter(event => Set("fruit_set", "pod_set").contains(event.eventType))
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
    val tallRecipientShortPollenLags = fruitEvents
      .filter(event => event.plantHeight == "tall" && event.pollenSourceHeight.contains("short"))
      .flatMap(_.pollinationToFruitTicks)
    val shortRecipientTallPollenLags = fruitEvents
      .filter(event => event.plantHeight == "short" && event.pollenSourceHeight.contains("tall"))
      .flatMap(_.pollinationToFruitTicks)
    val operatorSoilLots = soilLotIndexes.toArray.sortBy(_._1).map { case (id, lotIndex) =>
      val multipliers = soilMultipliers(requestedSoilNutrientNoise)
      "{" +
        "\"plant_id\":" + id + "," +
        "\"soil_lot_id\":\"soil-lot-" + lotIndex + "\"," +
        "\"soil_stage_multiplier\":" + multipliers(lotIndex) + "}"
    }

    "{" +
      "\"active\":" + enabled + "," +
      "\"world\":\"" + escapeJSON(requestedWorld) + "\"," +
      "\"mechanism\":\"" + escapeJSON(mechanismName) + "\"," +
      "\"case_root\":" + requestedRoot + "," +
      "\"episode_seed\":" + episodeSeed + "," +
      "\"preference_weight\":" + requestedPreferenceWeight + "," +
      "\"white_preference_weight\":" + requestedPreferenceWeight + "," +
      "\"soil_nutrient_lot_level\":" + requestedSoilNutrientNoise + "," +
      "\"fruit_set_success_level\":" + requestedFruitSetNoise + "," +
      "\"cross_parentage_contamination_level\":" + requestedContaminationNoise + "," +
      "\"growth_noise_level\":" + requestedSoilNutrientNoise + "," +
      "\"flower_count_noise_level\":" + requestedFruitSetNoise + "," +
      "\"fruit_timing_noise_level\":" + requestedContaminationNoise + "," +
      "\"soil_lot_assignments\":[" + operatorSoilLots.mkString(",") + "]," +
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
      "\"pod_sets\":" + fruitSets + "," +
      "\"fruit_sets\":" + fruitSets + "," +
      "\"white_fruit_sets\":" + whiteFruitSets + "," +
      "\"purple_fruit_sets\":" + purpleFruitSets + "," +
      "\"fertility_accelerated_ticks\":" + fertilityAcceleratedTicks + "," +
      "\"episode_tick\":" + episodeTick + "," +
      "\"white_pollination_ticks\":[" + whitePollinationTicks.mkString(",") + "]," +
      "\"purple_pollination_ticks\":[" + purplePollinationTicks.mkString(",") + "]," +
      "\"white_fruit_lags\":[" + whiteFruitLags.mkString(",") + "]," +
      "\"purple_fruit_lags\":[" + purpleFruitLags.mkString(",") + "]," +
      "\"tall_recipient_short_pollen_lags\":[" + tallRecipientShortPollenLags.mkString(",") + "]," +
      "\"short_recipient_tall_pollen_lags\":[" + shortRecipientTallPollenLags.mkString(",") + "]," +
      "\"initial_comparable_window_size\":" + initialWindow.length + "," +
      "\"initial_white_visits\":" + initialWindow.count(_.perceivedColor == "white") + "," +
      "\"initial_purple_visits\":" + initialWindow.count(_.perceivedColor == "purple") + "," +
      "\"transient_root_seed\":" + optionLongJSON(transientRootSeed) +
      ",\"clean_root_seed\":" + optionLongJSON(cleanRootSeed) +
      ",\"cross_attempts\":[" + crossAttempts.sortBy(_.attemptId).map { attempt =>
        "{" +
          "\"attempt_id\":\"cross-" + attempt.attemptId + "\"," +
          "\"flower_id\":" + attempt.flowerId + "," +
          "\"recipient_plant_id\":" + attempt.recipientPlantId + "," +
          "\"intended_pollen_plant_id\":" + attempt.intendedPollenPlantId + "," +
          "\"actual_pollen_plant_id\":" + attempt.actualPollenPlantId + "," +
          "\"contamination_occurred\":" + attempt.contaminationOccurred + "," +
          "\"status\":\"" + escapeJSON(attempt.status) + "\"," +
          "\"pod_id\":" + optionIntJSON(attempt.podId) + "}"
      }.mkString(",") + "]" +
      "}"
  }

  def publicStatusJSON(universe: EnvObject): String = {
    if (!enabled) return "{\"episode_tick\":0,\"pots\":[]}"
    val pots = universe.getContainedObjectsAndPortalsRecursive(includeHidden = false)
      .filter(obj => obj.isInstanceOf[FlowerPot] || obj.isInstanceOf[SelfWateringFlowerPot])
      .toArray
      .sortBy(obj => (obj.name, obj.uuid))
    val packed = pots.map { pot =>
      val direct = pot.getContainedObjects()
      val recursive = pot.getContainedObjectsRecursive()
      val planted = direct.collect { case plant: Plant => plant }.toArray.sortBy(_.uuid)
      val plantedIds = planted.map(_.uuid).toSet
      val stages = planted.map(plant => "\"" + escapeJSON(plant.lifecycle.get.getCurStageName()) + "\"")
      val flowers = recursive.collect { case flower: Flower => flower }
      val pending = flowers.count(_.propPollination.exists(_.pollinationStep > 0))
      val formedSeeds = recursive.collect { case plant: Plant if !plantedIds.contains(plant.uuid) => plant }.size
      val plantDetails = planted.map { plant =>
        val plantFlowers = plant.getContainedObjectsRecursive().collect { case flower: Flower => flower }
          .toArray.sortBy(_.uuid)
        "{" +
          "\"plant_id\":" + stableId(plantIds, plant.uuid) + "," +
          "\"stage\":\"" + escapeJSON(plant.lifecycle.get.getCurStageName()) + "\"," +
          "\"height\":\"" + escapeJSON(plantHeight(plant)) + "\"," +
          "\"soil_lot_id\":\"" + soilLotName(plant) + "\"," +
          "\"active_flowers\":[" + plantFlowers.map { flower =>
            "{" +
              "\"flower_id\":" + stableId(flowerIds, flower.uuid) + "," +
              "\"perceived_color\":\"" + escapeJSON(flower.getPerceivedColor) + "\"" +
              "}"
          }.mkString(",") + "]," +
          "\"active_flower_ids\":[" + plantFlowers.map { flower =>
            stableId(flowerIds, flower.uuid)
          }.mkString(",") + "]}"
      }
      val podCount = crossAttempts.count { attempt =>
        attempt.recipientPot == pot.name && attempt.status == "pod_set"
      }
      "{" +
        "\"name\":\"" + escapeJSON(pot.name) + "\"," +
        "\"has_soil\":" + direct.exists(_.isInstanceOf[Soil]) + "," +
        "\"has_water\":" + direct.exists(_.isInstanceOf[Water]) + "," +
        "\"plant_count\":" + planted.length + "," +
        "\"plant_stages\":[" + stages.mkString(",") + "]," +
        "\"plants\":[" + plantDetails.mkString(",") + "]," +
        "\"flower_count\":" + flowers.size + "," +
        "\"pending_fruit_count\":" + pending + "," +
        "\"formed_pod_count\":" + podCount + "," +
        "\"formed_seed_count\":" + formedSeeds +
        "}"
    }
    val seeds = universe.getContainedObjectsAndPortalsRecursive(includeHidden = false)
      .collect { case seed: Plant if seed.isSeed() => seed }
      .toArray
      .sortBy(_.uuid)
    val packedSeeds = seeds.map { seed =>
      val id = stableId(plantIds, seed.uuid)
      val container = seed.getContainer().map(_.name).getOrElse("uncontained")
      "{" +
        "\"seed_id\":\"seed-" + id + "\"," +
        "\"container\":\"" + escapeJSON(container) + "\"," +
        "\"maternal_plant_id\":" + optionIntJSON(seedMaternalPlantIds.get(seed.uuid)) + "," +
        "\"pod_id\":" + optionIntJSON(seedPodIds.get(seed.uuid)) + "," +
        "\"intended_pollen_plant_id\":" +
          optionIntJSON(seedIntendedPollenPlantIds.get(seed.uuid)) + "," +
        "\"source\":\"" + escapeJSON(seedPodIds.get(seed.uuid)
          .map(id => "pod-" + id + " from maternal plant-" + seedMaternalPlantIds(seed.uuid))
          .getOrElse("initial seed stock")) + "\"" +
        "}"
    }
    val publicReproduction = reproductionEvents.sortBy(_.index).map { event =>
      "{" +
        "\"event_type\":\"" + escapeJSON(event.eventType) + "\"," +
        "\"tick\":" + event.tick + "," +
        "\"flower_id\":" + event.flowerId + "," +
        "\"recipient_plant_id\":" + event.plantId + "," +
        "\"recipient_height\":\"" + escapeJSON(event.plantHeight) + "\"," +
        "\"flower_pot\":\"" + escapeJSON(event.flowerPot) + "\"," +
        "\"pollination_to_fruit_ticks\":" + optionIntJSON(event.pollinationToFruitTicks) +
        "}"
    }
    val publicAttempts = crossAttempts.sortBy(_.attemptId).map { attempt =>
      "{" +
        "\"attempt_id\":\"cross-" + attempt.attemptId + "\"," +
        "\"flower_id\":" + attempt.flowerId + "," +
        "\"recipient_plant_id\":" + attempt.recipientPlantId + "," +
        "\"recipient_pot\":\"" + escapeJSON(attempt.recipientPot) + "\"," +
        "\"intended_pollen_plant_id\":" + attempt.intendedPollenPlantId + "," +
        "\"intended_pollen_pot\":\"" + escapeJSON(attempt.intendedPollenPot) + "\"," +
        "\"emasculated\":" + attempt.emasculated + "," +
        "\"bagged\":" + attempt.bagged + "," +
        "\"started_tick\":" + attempt.startedTick + "," +
        "\"status\":\"" + escapeJSON(attempt.status) + "\"," +
        "\"resolved_tick\":" + optionIntJSON(attempt.resolvedTick) + "," +
        "\"pod_id\":" + optionIntJSON(attempt.podId) + "," +
        "\"seed_ids\":[" + attempt.seedIds.map(id => "\"" + escapeJSON(id) + "\"")
          .mkString(",") + "]}"
    }
    "{\"episode_tick\":" + episodeTick +
      ",\"pots\":[" + packed.mkString(",") + "]," +
      "\"seeds\":[" + packedSeeds.mkString(",") + "]," +
      "\"cross_attempts\":[" + publicAttempts.mkString(",") + "]," +
      "\"reproduction_history\":[" + publicReproduction.mkString(",") + "]}"
  }

  private def publicPots(universe: EnvObject): Array[EnvObject] = {
    universe.getContainedObjectsAndPortalsRecursive(includeHidden = false)
      .filter(obj => obj.isInstanceOf[FlowerPot] || obj.isInstanceOf[SelfWateringFlowerPot])
      .toArray
      .sortBy(obj => (obj.name, obj.uuid))
  }

  def controlledCrossJSON(
    universe: EnvObject,
    recipientPots: Array[String],
    donorPots: Array[String],
    emasculatedFlags: Array[Boolean],
    baggedFlags: Array[Boolean]
  ): String = {
    val lengths = Set(
      recipientPots.length, donorPots.length, emasculatedFlags.length, baggedFlags.length
    )
    if (lengths.size != 1 || recipientPots.isEmpty) {
      return "{\"ok\":false,\"error\":\"cross field counts differ or are empty\"}"
    }
    val potsByName = publicPots(universe).map(pot => pot.name -> pot).toMap
    val unknown = (recipientPots ++ donorPots).filterNot(potsByName.contains).distinct
    if (unknown.nonEmpty) {
      return "{\"ok\":false,\"error\":\"unknown flower pots: " +
        escapeJSON(unknown.mkString(", ")) + "\"}"
    }
    val allPlants = publicPots(universe).flatMap { pot =>
      pot.getContainedObjects().collect { case plant: Plant => plant }
    }.sortBy(_.uuid)
    val prepared = recipientPots.indices.map { index =>
      val recipientPot = potsByName(recipientPots(index))
      val donorPot = potsByName(donorPots(index))
      val recipients = recipientPot.getContainedObjects().collect { case plant: Plant => plant }
        .toArray.sortBy(_.uuid)
      val donors = donorPot.getContainedObjects().collect { case plant: Plant => plant }
        .toArray.sortBy(_.uuid)
      val flowers = recipientPot.getContainedObjectsRecursive().collect {
        case flower: Flower if flower.propPollination.exists(_.pollinationStep == 0) => flower
      }.toArray.sortBy(_.uuid)
      if (recipients.length != 1 || donors.length != 1 || flowers.isEmpty) {
        return "{\"ok\":false,\"error\":\"each recipient and donor pot must contain one plant, " +
          "and every recipient must have an unpollinated flower\"}"
      }
      (recipients.head, donors.head, flowers.head, emasculatedFlags(index), baggedFlags(index))
    }

    val results = prepared.map { case (recipient, intendedDonor, flower, emasculated, bagged) =>
      val stableFlowerId = stableId(flowerIds, flower.uuid)
      val rng = contaminationRNGs.getOrElseUpdate(
        stableFlowerId,
        new Random(episodeSeed.toLong ^ (stableFlowerId.toLong * 1000003L) ^ 0xC07A61A7L)
      )
      val contaminated = rng.nextDouble() < contaminationProbability(emasculated, bagged)
      val external = allPlants.find { plant =>
        plant.uuid != recipient.uuid && plant.uuid != intendedDonor.uuid
      }
      val actualDonor = if (!contaminated) intendedDonor
        else if (!emasculated) recipient
        else external.getOrElse(recipient)
      if (emasculated) {
        flower.getContainedObjects().collect { case pollen: Pollen => pollen }.foreach(_.delete())
      }
      val attemptId = crossAttempts.length
      val intendedId = stableId(plantIds, intendedDonor.uuid)
      val actualId = stableId(plantIds, actualDonor.uuid)
      flowerAttemptIds.update(stableFlowerId, attemptId)
      crossAttempts.append(AERCrossAttempt(
        attemptId = attemptId,
        flowerId = stableFlowerId,
        recipientPlantId = stableId(plantIds, recipient.uuid),
        recipientPot = plantPot(recipient),
        intendedPollenPlantId = intendedId,
        intendedPollenPot = plantPot(intendedDonor),
        actualPollenPlantId = actualId,
        emasculated = emasculated,
        bagged = bagged,
        contaminationOccurred = contaminated,
        startedTick = episodeTick,
        status = "pending",
        resolvedTick = None,
        podId = None,
        seedIds = Array.empty[String]
      ))
      val pollen = new Pollen(parentPlant = actualDonor)
      flower.addObject(pollen)
      val pollinated = flower.pollinateControlled(pollen, allowSelf = true)
      if (!pollinated) {
        crossAttempts(attemptId).status = "rejected"
        crossAttempts(attemptId).resolvedTick = Some(episodeTick)
      }
      "{" +
        "\"attempt_id\":\"cross-" + attemptId + "\"," +
        "\"recipient_pot\":\"" + escapeJSON(plantPot(recipient)) + "\"," +
        "\"recipient_plant_id\":" + stableId(plantIds, recipient.uuid) + "," +
        "\"intended_pollen_pot\":\"" + escapeJSON(plantPot(intendedDonor)) + "\"," +
        "\"intended_pollen_plant_id\":" + intendedId + "," +
        "\"emasculated\":" + emasculated + "," +
        "\"bagged\":" + bagged + "," +
        "\"status\":\"" + crossAttempts(attemptId).status + "\"}"
    }
    "{\"ok\":true,\"operation\":\"controlled-cross\",\"results\":[" +
      results.mkString(",") + "]}"
  }

  def batchWaterJSON(universe: EnvObject, targetNames: Array[String]): String = {
    val potsByName = publicPots(universe).map(pot => pot.name -> pot).toMap
    val unknown = targetNames.filterNot(potsByName.contains)
    if (unknown.nonEmpty) {
      return "{\"ok\":false,\"error\":\"unknown flower pots: " +
        escapeJSON(unknown.mkString(", ")) + "\"}"
    }
    val results = targetNames.map { name =>
      val pot = potsByName(name)
      val alreadyWet = pot.getContainedObjects().exists(_.isInstanceOf[Water])
      if (!alreadyWet) pot.addObject(new Water())
      "{\"pot\":\"" + escapeJSON(name) + "\",\"changed\":" + (!alreadyWet) + "}"
    }
    "{\"ok\":true,\"operation\":\"water\",\"results\":[" +
      results.mkString(",") + "]}"
  }

  def batchSowJSON(
    universe: EnvObject,
    seedIds: Array[String],
    targetNames: Array[String]
  ): String = {
    if (seedIds.length != targetNames.length) {
      return "{\"ok\":false,\"error\":\"seed and target counts differ\"}"
    }
    val potsByName = publicPots(universe).map(pot => pot.name -> pot).toMap
    val seeds = universe.getContainedObjectsAndPortalsRecursive(includeHidden = false)
      .collect { case seed: Plant if seed.isSeed() => seed }
      .toArray
      .sortBy(_.uuid)
    val seedsById = seeds.map { seed =>
      "seed-" + stableId(plantIds, seed.uuid) -> seed
    }.toMap
    val unknownPots = targetNames.filterNot(potsByName.contains)
    val unknownSeeds = seedIds.filterNot(seedsById.contains)
    val occupiedPots = targetNames.filter { name =>
      potsByName.get(name).exists(_.getContainedObjects().exists(_.isInstanceOf[Plant]))
    }
    if (unknownPots.nonEmpty || unknownSeeds.nonEmpty || occupiedPots.nonEmpty) {
      val details = Array(
        if (unknownPots.nonEmpty) Some("unknown pots: " + unknownPots.mkString(", ")) else None,
        if (unknownSeeds.nonEmpty) Some("unknown seeds: " + unknownSeeds.mkString(", ")) else None,
        if (occupiedPots.nonEmpty) Some("occupied pots: " + occupiedPots.mkString(", ")) else None
      ).flatten.mkString("; ")
      return "{\"ok\":false,\"error\":\"" + escapeJSON(details) + "\"}"
    }
    seedIds.zip(targetNames).foreach { case (seedId, targetName) =>
      potsByName(targetName).addObject(seedsById(seedId))
    }
    val results = seedIds.zip(targetNames).map { case (seedId, targetName) =>
      "{\"seed_id\":\"" + escapeJSON(seedId) + "\",\"pot\":\"" +
        escapeJSON(targetName) + "\"}"
    }
    "{\"ok\":true,\"operation\":\"sow\",\"results\":[" +
      results.mkString(",") + "]}"
  }
}
